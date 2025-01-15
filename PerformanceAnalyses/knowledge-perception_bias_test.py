from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import argparse
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=None, type=str)
    args = parser.parse_args()

    if args.size == "8B":
        nli_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir = "/data/hleier/MA/EvalModels"
        )
        nli_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            device_map="auto", 
            cache_dir = "/data/hleier/MA/EvalModels"
        )

    elif args.size == "70B":
        nli_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-70B-Instruct",
            cache_dir = "/data/hleier/MA/EvalModels"
        )
        nli_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-70B-Instruct",
            device_map="auto", 
            load_in_8bit=True,
            cache_dir = "/data/hleier/MA/EvalModels"
        )

    pipeline = pipeline(
        "text-generation",
        model=nli_model,
        tokenizer=nli_tokenizer,
    ) 

    # Hypothesis: First Location; Presmise: Unobserved, Observed
    first_location_pairs = [
        {
            "premise": "Muhammad picks up the remote control to turn on the television and changes the channel. Muhammad places the remote control in a small basket on the side table. While tidying up, Muhammad moves the remote control from the small basket to a drawer in the TV stand. Mariya excuses herself to go to the kitchen to prepare snacks. Mariya returns to the living room with snacks and drinks.",
            "hypo": "Mariya likely believes the remote control is in the small basket."
        },
        {
            "premise": "Muhammad picks up the remote control to turn on the television and changes the channel. Muhammad places the remote control in a small basket on the side table. Mariya excuses herself to go to the kitchen to prepare snacks. While tidying up, Muhammad moves the remote control from the small basket to a drawer in the TV stand. Mariya returns to the living room with snacks and drinks.",
            "hypo": "Mariya likely believes the remote control is in the small basket."
        },
        {
            "premise": "After using the stapler, Liam places it in the desk drawer. Liam then moves the stapler from the desk drawer to a stationery organizer on his desk. Lin steps out of the office to attend a meeting. Lin returns to the office after her meeting.",
            "hypo": "Lin likely believes the stapler is in the desk drawer."
        },
        {
            "premise": "After using the stapler, Liam places it in the desk drawer. Lin steps out of the office to attend a meeting. Liam then moves the stapler from the desk drawer to a stationery organizer on his desk. Lin returns to the office after her meeting.",
            "hypo": "Lin likely believes the stapler is in the desk drawer."
        },
        {
            "premise": "Maria interacts with a vintage record album. Isabella leaves the basement to grab cleaning supplies from upstairs. Isabella returns with the cleaning supplies. Then the record album is placed in a cardboard box. Maria moves the record album from the cardboard box to a plastic bin for more valuable items.",
            "hypo": "Isabella likely believes the record album is in the cardboard box."
        },
        {
            "premise": "Maria interacts with a vintage record album. Then the record album is placed in a cardboard box. Isabella leaves the basement to grab cleaning supplies from upstairs. Maria moves the record album from the cardboard box to a plastic bin for more valuable items. Isabella returns with the cleaning supplies.",
            "hypo": "Isabella likely believes the record album is in the cardboard box."
        },
        {
            "premise": "Ali picks up the mystery novel and flips through its pages. Ali places the mystery novel in a small basket beside the couch. Somsak excuses himself to use the restroom. Somsak returns to the living room after using the restroom. Ali moves the mystery novel from the small basket to a bookshelf near the window.",
            "hypo": "Somsak likely believes the mystery novel is in the small basket."
        },
        {
            "premise": "Ali picks up the mystery novel and flips through its pages. Ali places the mystery novel in a small basket beside the couch. Somsak excuses himself to use the restroom. Ali moves the mystery novel from the small basket to a bookshelf near the window. Somsak returns to the living room after using the restroom.",
            "hypo": "Somsak likely believes the mystery novel is in the small basket."
        }
    ]

    # Hypothesis: Second Location; Presmise: Unobserved, Observed
    second_location_pairs = [
        {
            "premise": "Muhammad picks up the remote control to turn on the television and changes the channel. Muhammad places the remote control in a small basket on the side table. While tidying up, Muhammad moves the remote control from the small basket to a drawer in the TV stand. Mariya excuses herself to go to the kitchen to prepare snacks. Mariya returns to the living room with snacks and drinks.",
            "hypo": "Mariya likely believes the remote control is in the drawer in the TV stand."
        },
        {
            "premise": "Muhammad picks up the remote control to turn on the television and changes the channel. Muhammad places the remote control in a small basket on the side table. Mariya excuses herself to go to the kitchen to prepare snacks. While tidying up, Muhammad moves the remote control from the small basket to a drawer in the TV stand. Mariya returns to the living room with snacks and drinks.",
            "hypo": "Mariya likely believes the remote control is in the drawer in the TV stand."
        },
        {
            "premise": "After using the stapler, Liam places it in the desk drawer. Liam then moves the stapler from the desk drawer to a stationery organizer on his desk. Lin steps out of the office to attend a meeting. Lin returns to the office after her meeting.",
            "hypo": "Lin likely believes the stapler is in the stationery organizer."
        },
        {
            "premise": "After using the stapler, Liam places it in the desk drawer. Lin steps out of the office to attend a meeting. Liam then moves the stapler from the desk drawer to a stationery organizer on his desk. Lin returns to the office after her meeting.",
            "hypo": "Lin likely believes the stapler is in the stationery organizer."
        },
        {
            "premise": "Maria interacts with a vintage record album. Isabella leaves the basement to grab cleaning supplies from upstairs. Isabella returns with the cleaning supplies. Then the record album is placed in a cardboard box. Maria moves the record album from the cardboard box to a plastic bin for more valuable items.",
            "hypo": "Isabella likely believes the record album is in the plastic bin."
        },
        {
            "premise": "Maria interacts with a vintage record album. Then the record album is placed in a cardboard box. Isabella leaves the basement to grab cleaning supplies from upstairs. Maria moves the record album from the cardboard box to a plastic bin for more valuable items. Isabella returns with the cleaning supplies.",
            "hypo": "Isabella likely believes the record album is in the plastic bin."
        },
        {
            "premise": "Ali picks up the mystery novel and flips through its pages. Ali places the mystery novel in a small basket beside the couch. Somsak excuses himself to use the restroom. Somsak returns to the living room after using the restroom. Ali moves the mystery novel from the small basket to a bookshelf near the window.",
            "hypo": "Somsak likely believes the mystery novel is in the bookshelf near the window."
        },
        {
            "premise": "Ali picks up the mystery novel and flips through its pages. Ali places the mystery novel in a small basket beside the couch. Somsak excuses himself to use the restroom. Ali moves the mystery novel from the small basket to a bookshelf near the window. Somsak returns to the living room after using the restroom.",
            "hypo": "Somsak likely believes the mystery novel is in the bookshelf near the window."
        }
    ]

    inference_dict = {
        "First Location": first_location_pairs,
        "Second Location": second_location_pairs
    }

    for k in inference_dict.keys():
        
        print(f"\n > Answer Location: {k}")

        for i, pair in enumerate(inference_dict[k]):
        
            print(f"\nPair {i}")
            print(f"Premise: {pair['premise']}")
            print(f"Hypothesis: {pair['hypo']}")

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