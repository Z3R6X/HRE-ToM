import transformers
import argparse
import time
import os
import json
import importlib
from tqdm import tqdm

from util.data import create_directory, transform_list_to_dataset


def load_tasklib(task_name):
    task = importlib.import_module("tasks." + task_name)
    return task


def create_cot_examples(json_file, num_examples, tasklib, prompt_template):
    
    # Load the CoT file
    data_list = json.load(open(json_file))

    if len(data_list) > num_examples:
        print("Not enough examples in data file, all examples used")

    messages = []
    # Add CoT examples to the chat template messages
    for i in range(min(num_examples, len(data_list))):
        
        # Insert the scenario in the task-specific and the general prompt template
        question = tasklib.get_question(data_list[i])
        prompt = prompt_template.format(data_list[i]["scenario"], question)

        # Append the example sceanario with question and the example response
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": data_list[i]["answer"]})

    return messages


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_kwargs", type=str, default=None)
    parser.add_argument("--tasks", nargs='+', default=None)
    parser.add_argument("--working_dir", default="Evaluation/outputs", type=str)
    parser.add_argument("--cache_dir", default="/data/hleier/MA/", type=str)
    parser.add_argument("--inference_settings", type=str, default="Evaluation/inference_settings.json")
    args = parser.parse_args()
    
    task_list = args.tasks

    time_id = time.strftime("%m.%d-%H:%M:%S")
    
    # Create run directory to store model outputs
    run_name = "run_" + time_id + "_" + args.model.replace("/", "-")
    run_path = os.path.join(args.working_dir, run_name)
    prediction_path = os.path.join(run_path, "predictions")
    create_directory(prediction_path)

    # Load inference settings from JSON file
    inference_settings = json.load(open(args.inference_settings))
    generation_args = inference_settings["generation_args"]
    prompt_template = inference_settings["prompt_template"]

    # Overwrite cache_dir variable in keyword arguments
    if args.model_kwargs is not None:
        kwargs = args.model_kwargs
    else:
        kwargs = {}
    kwargs["cache_dir"] = args.cache_dir

    # Load model pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        device_map="auto",
        model_kwargs=kwargs,
    )

    # Create the info dict for the run
    run_info = {
        "model": {
            "hf_model_id": args.model,
            "hf_model_kwargs": kwargs,
            "inferece_settings": inference_settings,
            "terminators": pipeline.tokenizer.eos_token_id
        },
    }  

    # Test model on all specified tasks
    for task_name in task_list:

        print(f"\nRunning inference on {task_name} task:")

        # Load task file
        tasklib = load_tasklib(task_name)

        # Save task info in the info dict
        task_dict = {
            "question_template": tasklib.get_question_template(),
            "data_file": tasklib.get_data_file(),
            "cot_file": tasklib.get_cot_file()      
        }              
        run_info[task_name] = task_dict

        # Load the data file and transform it into a dataset
        data_list = json.load(open(tasklib.get_data_file()))
        dataset = transform_list_to_dataset(data_list)

        # Create CoT examples if specified
        if inference_settings["num_cot"]:
            cot_messages = create_cot_examples(
                json_file= tasklib.get_cot_file(),
                num_examples=inference_settings["num_cot"],
                tasklib = tasklib,
                prompt_template = prompt_template
            )     
        else:
           cot_messages = []

        # Define terminators for text generation
        terminators = [
            pipeline.tokenizer.eos_token_id,
        ]

        prediction_list = []

        for i, row in tqdm(enumerate(dataset), total=len(dataset)):
            
            # Insert the scenario in the task-specific and the general prompt template
            question = tasklib.get_question(row)
            prompt = prompt_template.format(row["scenario"], question)

            # Insert the prompt in the chat template
            system_message = [{"role": "system", "content": inference_settings["system_prompt"]}] 
            all_messages = system_message + cot_messages + [{"role": "user", "content": prompt}]
            full_prompt = pipeline.tokenizer.apply_chat_template(
                all_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            #print(f"Prompt: {full_prompt}")

            # Generate model output
            outputs = pipeline(
                full_prompt,
                max_new_tokens=generation_args["max_new_tokens"],
                eos_token_id=terminators,
                do_sample=bool(generation_args["do_sample"]),
                temperature=generation_args["temperature"],
                top_p=generation_args["top_p"],
            )
            prediction = outputs[0]["generated_text"][len(full_prompt):]
            prediction_dict = {}

            for key in row.keys():
                prediction_dict[key] = row[key]
            prediction_dict["prediction"] = prediction
            prediction_list.append(prediction_dict)

        # Save the generated responses in the run directory
        results = json.dumps(prediction_list, indent=4)
        with open(os.path.join(prediction_path, task_name+".json"), "w") as outfile:
            outfile.write(results)

    
    # Save the run info in the run directory 
    run_info = json.dumps(run_info, indent=4)
    with open(os.path.join(run_path, "run_info.json"), "w") as outfile:
        outfile.write(run_info)

    print(f"Results stored in: {run_path}")


if __name__ == "__main__":
    main()
