import openai
import os
import json
import argparse
import random


def generate_scenario(setting, names, num_samples=1):

    # Prompt template with basic sequence of events and instruction
    prompt = f"""
Here is a sequence of events:
1. PersonX and PersonY are in the {setting}.
2. PersonX interacts with OBJECT.
3. OBJECT is placed in CONTAINER1.
4. PersonY leaves the SETTING.
5. PersonX moves the OBJECT from CONTAINER1 to CONTAINER2.
6. PersonX performs a task in the SETTING.
7. PersonY returns to the SETTING.
8. PersonX performs a task in the SETTING.

Please write a scenario based on this sequence. Replace CONTAINER1 and CONTAINER2 with containers that fit the rest of the scenario. Replace the generic events with a specific action that fits with the rest of the scenario.
"""
    
    # Replace placeholder names with two random names
    sampled_names = random.sample(names, k=2)
    prompt = prompt.replace("PersonX", sampled_names[0]).replace("PersonY", sampled_names[1])

    # Insert prompt into chat template
    messages = [
        {"role": "system", "content": "You are an assistant that generates simple, factual social interaction scenarios for psychological research."},
        {"role": "user", "content": prompt}
    ]

    # Generate response with OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=200,
        temperature=0.6,
        n=num_samples,
    )

    scenario = {"generated_scenario": c.message.content for i, c in enumerate(response.choices)}
    
    return scenario
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--basic_scenarios", type=str, default="basic_scenarios.json")
    parser.add_argument("--names", type=str, default="common_names.json")
    args = parser.parse_args()

    # Insert OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Load basic settings from file
    basic_settings = json.load(open(args.basic_settings))
    
    num_samples = args.num_samples

    # Load names from file
    names = json.load(open(args.names))

    generated_scenarios = []

    # Generate full scenario for each basic setting
    for i, basic_setting in enumerate(basic_settings):
        
        response = generate_scenario(
            basic_setting=basic_setting,
            names=names,
            num_samples=num_samples
        )

        scenario_dict = {
            "situation_id":  basic_setting["situation_id"],
            "setting":  basic_setting["setting"],
            "response": response
        }

        generated_scenarios.append(scenario_dict)

    # Save responses
    output = json.dumps(generated_scenarios, indent=4)
    with open("KP_responses.json", "w") as outfile:
        outfile.write(output)
