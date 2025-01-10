from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import T5ForConditionalGeneration
from transformers import AutoModelForCausalLM
from transformers import pipeline
import argparse
import torch
from scipy.special import softmax
import torch.nn.functional as F
import transformers


# Not used in test
def get_t5_logprob(prompt, model, tokenizer):

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate scores
    outputs = model.generate(
        input_ids,
        output_scores=True,
        return_dict_in_generate=True,
    )
    result = int(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
    print(f"Result: {result}")

    if result == 0:
        assert outputs.scores[1].argmax() == 632, "Wrong 0 token index"

    elif result == 1:
        assert outputs.scores[0].argmax() == 209, "Wrong 1 token index"

    else:
        raise Exception

    # Get entailment and contradiction probabilities
    contra = outputs.scores[1][0][632].item()
    entail = outputs.scores[0][0][209].item()

    return softmax([contra, entail])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--nli_model", default=None, type=str)
    args = parser.parse_args()

    if args.nli_model == "deberta":

        nli_tokenizer = AutoTokenizer.from_pretrained(
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", 
            cache_dir = "/data/hleier/MA/EvalModels"
        )
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            cache_dir = "/data/hleier/MA/EvalModels"
        )

    elif args.nli_model =="t5":
        
        nli_tokenizer = AutoTokenizer.from_pretrained(
            "google/t5_xxl_true_nli_mixture",
            cache_dir="/data/hleier/MA/EvalModels"
        )
        nli_model = T5ForConditionalGeneration.from_pretrained(
            "google/t5_xxl_true_nli_mixture",
            device_map="auto",
            cache_dir="/data/hleier/MA/EvalModels"
        )

    elif args.nli_model =="llama":
        nli_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir = "/data/hleier/MA/EvalModels"
        )
        nli_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            device_map="auto", 
            cache_dir = "/data/hleier/MA/EvalModels"
        )

        pipeline = transformers.pipeline(
            "text-generation",
            model=nli_model,
            tokenizer=nli_tokenizer,
        ) 
        
    print(nli_model.num_parameters())
    
    # Deductive premise-hypothesis pairs, not used in test
    deductive_inference_pairs = [
        {
            "premise": "All planets in our solar system orbit the Sun. Earth is a planet in our solar system.",
            "hypo": "Earth orbits the Sun."
        },
        {
            "premise": "All planets in our solar system orbit the Sun. Europa is not a planet in our solar system.",
            "hypo": "Europa orbits the Sun."
        },
        {
            "premise": "If it rains, the ground will get wet. It is raining.",
            "hypo": "The ground will get wet."
        },
        {
            "premise": "If it rains, the ground will get wet. It is not raining.",
            "hypo": "The ground will get wet."
        },
        {
            "premise": "Every even number greater than 2 can be expressed as the sum of two prime numbers. 10 is an even number greater than 2.",
            "hypo": "10 can be expressed as the sum of two prime numbers."
        },
        {
            "premise": "Every even number greater than 2 can be expressed as the sum of two prime numbers. 7 is a odd number greater than 2.",
            "hypo": "7 can be expressed as the sum of two prime numbers."
        }
    ]

    # Non-Social-Inductive premise-hypothesis pairs, deterministic and non-deterministic
    non_deductive_inference_pairs = [
        {
            "premise": "All swans observed so far are white. A new swan is observed.",
            "hypo": "The new swan is likely white."
        },
        {
            "premise": "All swans observed so far are white. A new swan is observed.",
            "hypo": "The new swan is white."
        },
        {
            "premise": "The past 10 winters have been colder than average.",
            "hypo": "The next winter will probably be colder than average."
        },
        {
            "premise": "The past 10 winters have been colder than average.",
            "hypo": "The next winter will be colder than average."
        },
        {
            "premise": "Students who consistently attend lectures tend to score higher on exams. Sarah attends all her lectures.",
            "hypo": "Sarah is likely to score higher on her exams."
        },
        {
            "premise": "Students who consistently attend lectures tend to score higher on exams. Sarah attends all her lectures.",
            "hypo": "Sarah will score higher on her exams."
        },
        {
            "premise": "In previous studies, high-protein diets have been linked to increased muscle growth. Emma started a high-protein diet.",
            "hypo": "Emma is likely to experience increased muscle growth."
        },
        {
            "premise": "In previous studies, high-protein diets have been linked to increased muscle growth. Emma started a high-protein diet.",
            "hypo": "Emma will experience increased muscle growth."
        },
    ]

    # Correct social-inductive premise-hypothesis pairs, deterministic and non-deterministic
    social_inference_pairs = [
        {
            "premise": "Sarah failed the exam.",
            "hypo": "Sarah feels sad."
        },
        {
            "premise": "Sarah failed the exam.",
            "hypo": "Sarah likely feels sad."
        },
        {
            "premise": "Alex gets a gift from her friend.",
            "hypo": "Alex is happy."
        },
        {
            "premise": "Alex gets a gift from her friend.",
            "hypo": "Alex might be happy."
        },
        {
            "premise": "Eva is not invited to the party.",
            "hypo": "Eva is disappointed."
        },
        {
            "premise": "Eva is not invited to the party.",
            "hypo": "Eva is probably disappointed."
        },
        {
            "premise": "Steve breaks up with Nora.",
            "hypo": "Nora is sad."
        },
        {
            "premise": "Steve breaks up with Nora.",
            "hypo": "Nora might be sad."
        },
        {
            "premise": "Max is offended by Sarahs comments.",
            "hypo": "Max is angry."
        },
        {
            "premise": "Max is offended by Sarahs comments.",
            "hypo": "Max might be angry."
        }
    ]

    # Incorrect social-inductive premise-hypothesis pairs, deterministic and non-deterministic
    wrong_social_inference_pairs = [
        {
            "premise": "Sarah failed the exam.",
            "hypo": "Sarah feels happy."
        },
        {
            "premise": "Sarah failed the exam.",
            "hypo": "Sarah likely feels happy."
        },
        {
            "premise": "Alex gets a gift from her friend.",
            "hypo": "Alex is sad."
        },
        {
            "premise": "Alex gets a gift from her friend.",
            "hypo": "Alex might be sad."
        },
        {
            "premise": "Eva is not invited to the party.",
            "hypo": "Eva is reliefed."
        },
        {
            "premise": "Eva is not invited to the party.",
            "hypo": "Eva is probably reliefed."
        },
        {
            "premise": "Steve breaks up with Nora.",
            "hypo": "Nora is joyful."
        },
        {
            "premise": "Steve breaks up with Nora.",
            "hypo": "Nora might be joyful."
        },
        {
            "premise": "Max is offended by Sarahs comments.",
            "hypo": "Max is cheerful."
        },
        {
            "premise": "Max is offended by Sarahs comments.",
            "hypo": "Max might be cheerful."
        }
    ]

    inference_dict = {
        "Deductive": deductive_inference_pairs,
        "Non-Deductive": non_deductive_inference_pairs,
        "Social (correct)": social_inference_pairs,
        "Social (incorrect)": wrong_social_inference_pairs
    }

    for k in inference_dict.keys():
        
        print(f"\n > Type of Reasoning: {k}")

        for pair in inference_dict[k]:

            print(f"\nPremise: {pair['premise']}")
            print(f"Hypothesis: {pair['hypo']}")

            if args.nli_model == "deberta":
                
                # Tokenize input
                input = nli_tokenizer(
                    pair["premise"], pair["hypo"], truncation=True, padding=True, return_tensors="pt"
                )
                
                # Get prediction
                with torch.no_grad():
                    output = nli_model(**input)
                prediction = torch.softmax(output["logits"], -1).tolist()
                
                # Print answer probability
                print(f"Entailment: {prediction[0][0]} - Neutral: {prediction[0][1]} - Contradiction: {prediction[0][2]}")

            elif args.nli_model == "t5":

                # Format prompt template for T5
                input_text = "premise: {} hypothesis: {}".format(pair["premise"], pair["hypo"])
                
                # Tokenize input
                input_ids = nli_tokenizer(input_text, return_tensors="pt").input_ids.to(nli_model.device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = nli_model.generate(input_ids, max_new_tokens=10)
                result = nli_tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = "Entailment" if result == "1" else "No Entailment"

                # Print classification result
                print(f"Classification: {prediction}")
                
            elif args.nli_model == "llama":

                # Get answer probabilities for the premise-hypothesis pair
                answer_probs = get_entailment_prob(
                    premise=pair["premise"],
                    conclusion=pair["hypo"],
                    pipeline=pipeline,
                    model=nli_model,
                    tokenizer=nli_tokenizer
                )

                # Print answer probabilities
                probs = answer_probs.items()
                print(probs)


# Same function as in the LLogNet class
def get_entailment_prob(premise, conclusion, pipeline, model, tokenizer):

        # NLI-prompt with CoT examples
        instruction = """Given a premise and a hypothesis, determine if the hypothesis follows logically from the premise.
        Choose one of the following options:
        - Entailment: if the hypothesis is directly supported by the premise.
        - Neutral: if the hypothesis is neither clearly supported nor contradicted by the premise.
        - Contradiction: if the hypothesis is contradicted by the premise.
        """
        instruction_question = "Does the hypothesis follow logically from the premise? (Entailment/Neutral/Contradiction)"

        input_text = f"{instruction}\nPremise: {premise}\nHypothesis: {conclusion}\n{instruction_question}"

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"{instruction}\nPremise: The bakery sells freshly baked bread every morning.\nHypothesis: You can buy fresh bread at the bakery in the morning\n{instruction_question}"},
            {"role": "assistant", "content": "Entailment"},
            {"role": "user", "content": f"{instruction}\nPremise: The bakery sells freshly baked bread every morning.\nHypothesis: The bakery is open until 8 p.m.\n{instruction_question}"},
            {"role": "assistant", "content": "Neutral"},
            {"role": "user", "content": f"{instruction}\nPremise: The bakery sells freshly baked bread every morning.\nHypothesis: The bakery is closed in the mornings.\n{instruction_question}"},
            {"role": "assistant", "content": "Contradiction"},
            {"role": "user", "content": input_text},
        ]

        input_msg = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )  
        
        # Define answer choices
        answers = ["Entailment", "Neutral", "Contradiction"]

        # Tokenize the NLI prompt
        prompt_inputs = tokenizer(input_msg, return_tensors="pt")
        prompt_ids = prompt_inputs.input_ids

        # Dictionary to store probabilities for each answer
        answer_probs = {}

        for answer in answers:
            # Tokenize the answer independently
            answer_inputs = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
            answer_ids = answer_inputs.input_ids

            # Concatenate prompt and answer ids
            input_ids = torch.cat([prompt_ids, answer_ids], dim=-1)

            # Store inverse perplexity for each answer
            answer_probs[answer] = get_inv_perplexity(input_ids, answer_ids, model=model)

        # Normalize to get probabilities for each answer
        total_prob = sum(answer_probs.values())
        for answer in answers:
            answer_probs[answer] /= total_prob

        return answer_probs


def get_inv_perplexity(input_ids, answer_ids, model):
        
        # Compute log probabilities with the model
        with torch.no_grad():
            outputs = model(input_ids)
            log_probs = F.log_softmax(outputs.logits, dim=-1)

        # Sum log probabilities
        answer_log_prob = sum(log_probs[0, -(1 + len(answer_ids[0])) + i, token_id].item() for i, token_id in enumerate(answer_ids[0]))

        # Adjust for token length
        average_log_prob = answer_log_prob / len(answer_ids[0])
        
        # Compute perplexity
        perplexity = torch.exp(-torch.tensor(average_log_prob))  # Adjusted perplexity

        # Return inverse probability
        return 1 / perplexity.item()


if __name__ == "__main__":
    main()