from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from tqdm import tqdm
#import numpy as np
from collections import defaultdict, deque

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util')))
from util.eval import split_all_chains, split_chain, calculate_means


class LLogNet:
    def __init__(
            self,
            device_map="auto",
            cache_dir="/data/hleier/MA/EvalModels",
            load_in_8bit=False,
            load_in_4bit=False,
            apply_splitting=False,
            verbose=2,
            return_full_scores=False
        ):

        self.name = "llognet"
        #self.device_map = device_map
        self.verbose = verbose
        self.apply_splitting = apply_splitting
        self.return_full_scores = return_full_scores

        # Overwrite load_in_8bit if load_in_4bit is set
        if load_in_4bit:
            load_in_8bit = False

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            device_map=device_map, 
            cache_dir=cache_dir,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir=cache_dir
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )   


    def load_predictions(self, predictions):
        
        # Load source texts, reasoning chains and character name
        print("\nLoad Sources and Predictions")
        sources = [d['scenario'] for d in predictions]
        reasoning_chains = [d['prediction'] for d in predictions]
        self.name_list = [d['question_name'] for d in predictions]

        # Determine task type (Knowledge-Perception task contains "object")
        if "object" in predictions[0].keys():
            self.task_type = "KP"
        else:
            self.task_type = "IE"

        # For Knowledge-Perception task, load objects
        if self.task_type == "KP":
            self.object_list = [d['object'] for d in predictions]

        self.num_responses = len(reasoning_chains)
        self.sources = sources
        self.reasoning_chains = reasoning_chains
        
        # Split source texts into steps
        self.sources_split = split_all_chains(sources)

        # Apply naive or semantic splitting (Step 1)
        if self.apply_splitting:
            self.reasoning_chains_split = self.transform_chains(reasoning_chains)
        else:
            self.reasoning_chains_split = split_all_chains(reasoning_chains)


    def transform_chains(self, reasoning_chains):
        
        ### Step 1
        print("\nStep 1 - Transform Reasoning Chains") if self.verbose >= 0 else None

        transformed_reasoning_chain_list = []

        for i, reasoning_chain in tqdm(enumerate(reasoning_chains), total=len(reasoning_chains)):
            
            # Insert reasoning chain in the prompt template
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"Split the following sentences into separate, standalone sentences where possible. If a sentence contains a comma that could indicate the start of a new sentence, separate it accordingly:\n{reasoning_chain}"} 
            ]
      
            generation_args = {
                "max_new_tokens": 250,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }

            # Apply chat template
            input_msg = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) 

            # Add answer prefix
            input_msg = input_msg + "Here are the sentences split into separate, standalone sentences:\n\n1."

            # Generate response
            output = self.pipeline(input_msg, **generation_args)
            text = output[0]['generated_text']
            print(f"\nResult: {text}") if self.verbose >= 2 else None

            # Apply naive splitting to result
            split_text = split_chain(text) 

            transformed_reasoning_chain_list.append(split_text)

        return transformed_reasoning_chain_list
    

    def evaluate_all(self):
        print("\nEvaluate LLogNet") 
        # Get results for:
        # Source Grounding (sg), 
        # Step Entailment (se) and 
        # Chain Entailment (ce)
        sg, se, ce = self.evaluate_llognet()

        # Save results 
        full_dict = {
            "source_grounding": sg,
            "step_entailment": se,
            "chain_entailment": ce 
        }

        return full_dict


    def evaluate_llognet(self):
        
        sg_result_list = []
        se_result_list = []
        ce_result_list = []

        print(f"Task Type: {self.task_type}") if self.verbose >= 1 else None

        for rc_id, (reasoning_chain, name) in tqdm(enumerate(zip(self.reasoning_chains_split, self.name_list)), total=self.num_responses):
            
            print(f"\n\n> TASK NR {rc_id}") if self.verbose >= 1 else None
            
            ### Step 2
            print("\nStep 2 - Determine Premises and Conclusions") if self.verbose >= 1 else None

            # Initialize lists to store premises and conclusions
            conclusion_list = []
            premises_list = []

            for i, rs in enumerate(reasoning_chain):
                
                print(f"\nReasoning Step: {rs}") if self.verbose >= 2 else None

                # Last step is always set to conlusion
                if i == len(reasoning_chain)-1:
                    print("> Last Element, therefore Conclusion") if self.verbose >= 2 else None
                    conclusion_list.append({"id": i, "conclusion": rs})  
                    break 

                if self.task_type == "IE":
                    
                    # CoT messages to classifiy Inferring Emotion reasoning steps
                    messages = [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": f"\"Michael's fast pace suggests caution\"\nDoes the sentence specify Michael’s emotion? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"Sofija’s facial expression tightens, and she clenches her jaw\"\nDoes the sentence specify Sofija’s emotion? (Yes/No)"},
                        {"role": "assistant", "content": "No"},
                        {"role": "user", "content": f"\"Fatma's raises her eyebrows and slightly opens her mouth\"\nDoes the sentence specify Fatma’s emotion? (Yes/No)"},
                        {"role": "assistant", "content": "No"},
                        {"role": "user", "content": f"\"Clara’s strong demeanor suggests anger\"\nDoes the sentence specify Clara’s emotion? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"Taking a photo often reflects excitement or fascination\"Does the sentence specify John’s emotion? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"This implies that Yuri’s emotion is noticeable and significant enough to draw Sophie’s attention\"\nDoes the sentence specify Yuri’s emotion? (Yes/No)"},
                        {"role": "assistant", "content": "No"},
                        {"role": "user", "content": f"\"The forceful action and harsh tone point to strong emotional involvement\"\nDoes the sentence specify Philipp’s emotion? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"Sergei clenches his fists and raises them\"\nDoes the sentence specify Sergei’s emotion? (Yes/No)"},
                        {"role": "assistant", "content": "No"},
                        {"role": "user", "content": f"\"Dragan's willingness to share his work suggests he is confident in its correctness\"\nDoes the sentence specify Dragan’s emotion? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"{rs}\"\nDoes the sentence specify {name}’s emotion? (Yes/No)"},
                    ]

                if self.task_type == "KP":
                    # CoT messages to classifiy Knowledge-Perception reasoning steps
                    messages = [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": f"\"Sarah has seen the cake in the fridge and is not aware of any changes\"\nDoes the sentence specify Clara’s knowledge about the cake? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"There is a cardboard box in the basement that contains books\"\nDoes the sentence specify Michael’s knowledge about that books? (Yes/No)"},
                        {"role": "assistant", "content": "No"},
                        {"role": "user", "content": f"\"Markus moves the knife from the dishwasher to the drawer.\"\nDoes the sentence specify Philipp’s knowledge about the knife? (Yes/No)"},
                        {"role": "assistant", "content": "No"},
                        {"role": "user", "content": f"\"Daniel was present when Fabian moved the phone from the box to the desk drawer\"\nDoes the sentence specify Daniel’s knowledge about the phone? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"Helen is still present when Victor puts the bottle in the trash and likely sees where it is placed\"\nDoes the sentence specify Helen’s knowledge about the bottle? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"Alex places the pillow in a basket beside the wardrobe\"\nDoes the sentence specify Morgan’s knowledge about the pillow? (Yes/No)"},
                        {"role": "assistant", "content": "No"},
                        {"role": "user", "content": f"\"Clara is present when Maya moves the orange to the basket\"\nDoes the sentence specify Clara’s knowledge about the orange? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"Bryan moves the watch from the drawer to a shelf near the stairs\"\nDoes the sentence specify Ibrahim’s knowledge about the watch? (Yes/No)"},
                        {"role": "assistant", "content": "No"},
                        {"role": "user", "content": f"\"Khai witnessed the movement of the notebook, so he would likely believe that the notebook is now in the backpack\"\nDoes the sentence specify Khai’s knowledge about the notebook? (Yes/No)"},
                        {"role": "assistant", "content": "Yes"},
                        {"role": "user", "content": f"\"{rs}\"\nDoes the sentence specify {name}’s knowledge about the {self.object_list[rc_id]}? (Yes/No)"},
                    ]

                generation_args = {
                    "max_new_tokens": 10,
                    "return_full_text": False,
                    "temperature": 0.0,
                    "do_sample": False,
                }

                # Generate response 
                output = self.pipeline(messages, **generation_args)
                text = output[0]['generated_text']

                # Classify reasoning step by response
                if text[:2] == "No":
                    print("> Likely Premise") if self.verbose >= 2 else None
                    premises_list.append({"id": i, "premise": rs})
                else:
                    print("> Likely Conclusion") if self.verbose >= 2 else None
                    conclusion_list.append({"id": i, "conclusion": rs})   

            print(f"\nConclusion List: {conclusion_list}") if self.verbose >= 2 else None
            print(f"\nPremise List: {premises_list}") if self.verbose >= 2 else None

            ### Step 3 - Premises
            print("\nStep 3 - Check Source to Premise Entailment") if self.verbose >= 1 else None

            # Initialize lists to save results for Source Grounding and Chain Entailment
            agreement_dicts = []
            premise_agreement_probs = []

            for premise in premises_list:

                # Get answer probabilities for each premise
                answer_probs = self.get_entailment_prob(premise=self.sources[rc_id], conclusion=premise['premise'])
                
                # Save non-contradiction probability
                premise_agreement_prob = answer_probs["Entailment"] + answer_probs["Neutral"]
                premise_agreement_probs.append(premise_agreement_prob)
                agreement_dicts.append({"node": premise["id"], "type": "premise", "agreement_prob": premise_agreement_prob})
                premise["src_entailment"] = premise_agreement_prob

            # Get mean and min premise aggreement probability
            avg_premise_agreement_prob = sum(premise_agreement_probs) / len(premise_agreement_probs)
            min_premise_agreement_prob = min(premise_agreement_probs)
            
            print("\nSource Grounding (Premises)") if self.verbose >= 2 else None
            print(f"Mean Premise Agreement Prob: {avg_premise_agreement_prob}") if self.verbose >= 2 else None
            print(f"Minimal Premise Agreement Prob: {min_premise_agreement_prob}") if self.verbose >= 2 else None

            ### Step 3 - Conclusions
            print("\nStep 3 - Check Source to Conclusion Entailment") if self.verbose >= 1 else None

            conclusion_agreement_probs = []

            for conc in conclusion_list:

                # Get answer probabilities for each conclusion
                answer_probs = self.get_entailment_prob(premise=self.sources[rc_id], conclusion=conc['conclusion'])

                # Save non-contradiction probability
                conclusion_agreement_prob = answer_probs["Entailment"] + answer_probs["Neutral"]
                conclusion_agreement_probs.append(conclusion_agreement_prob)
                agreement_dicts.append({"node": conc["id"], "type": "conclusion", "agreement_prob": conclusion_agreement_prob})
            
            # Get mean and min conclusion aggreement probability
            avg_conclusion_agreement_prob = sum(conclusion_agreement_probs) / len(conclusion_agreement_probs)
            min_conclusion_agreement_prob = min(conclusion_agreement_probs)

            print("\nSource Grounding (Conclusions)") if self.verbose >= 2 else None
            print(f"Mean Conclusion Agreement Prob: {avg_conclusion_agreement_prob}") if self.verbose >= 2 else None
            print(f"Minimal Conclusion Agreement Prob: {min_conclusion_agreement_prob}") if self.verbose >= 2 else None

            # Save results
            sg_result = {
                "id": rc_id,
                "premise_agreement (min)": min_premise_agreement_prob, 
                "conclusion_agreement (min)": min_conclusion_agreement_prob,
                "premise_agreement (mean)": avg_premise_agreement_prob, 
                "conclusion_agreement (mean)": avg_conclusion_agreement_prob
            }
            
            ### Step 4
            print("\nStep 4 - Check Entailment") if self.verbose >= 1 else None
            
            # Initialize ID for previous conclusion
            prev_id = 0

            # Initialize list to store previous conclusion
            prev_concs = [] 

            #contradiction_probs = []
            # Initialize lists to store probabilities for Step Entailment
            non_contradiction_probs = []
            entailment_probs = []

            # Initialize list to store the graph structure and probabilities for Chain Entailment 
            entailment_dicts = []

            #max_probs = []
            #mean_probs = []

            # Initialite edge ID
            edge_id = 0

            #src_entailment = []

            for conc in conclusion_list:

                print(f"\nConclusion {conc['id']}: {conc['conclusion']}") if self.verbose >= 2 else None

                print("\nPremise to Conlusion Entailment") if self.verbose >= 2 else None

                # Get all premises preceeding the currect conclusion
                premises = [p for p in premises_list if p["id"] < conc["id"] and p["id"] >= prev_id]

                #prem_to_conc_probs = []

                # Entailment from preceeding premises to currect conclusion
                if premises:

                    # Join premises to one string
                    joined_premise = ". ".join([p["premise"] for p in premises])

                    print(f"\nPremises: {joined_premise}") if self.verbose >= 2 else None

                    # Get node ID from premise ID
                    node_id = premises[0]["id"]

                    #avg_joined_prob = sum([p["src_entailment"] for p in premises]) / len(premises)
                    #src_entailment.append({"from": -1, "to": node_id, "ent_prob": avg_joined_prob, "ncont_prob": avg_joined_prob})
                    
                    # Get answer probabilities for each conclusion
                    answer_probs = self.get_entailment_prob(premise=joined_premise, conclusion=conc['conclusion'])

                    # Save entailment probability
                    entailment_prob = answer_probs["Entailment"]
                    entailment_probs.append(entailment_prob)

                    # Save non-contradiction probability
                    non_contradiction_prob = answer_probs["Entailment"] + answer_probs["Neutral"]
                    non_contradiction_probs.append(non_contradiction_prob)

                    # Update entailment dict with ID of outgoing and incoming node and probabilities
                    entailment_dicts.append({"id": edge_id, "from": node_id, "to": conc["id"], "ent_prob": entailment_prob, "ncont_prob": non_contradiction_prob})
                    edge_id +=1

                # Conc to Conc Entailment
                print("\nPrevious Conclusion to Current Conlusion Entailment") if self.verbose >= 2 else None
                
                #conc_to_conc_probs = []

                #conc_to_conc_probs_mean = []
                #conc_to_conc_probs_max = []

                # Entailment from previous conclusion to current conclusion 
                for prev_conc in prev_concs:

                    print(f"\nPrev Conclusion: {prev_conc}") if self.verbose >= 2 else None

                    # Get answer probabilities for each conclusion
                    answer_probs = self.get_entailment_prob(premise=prev_conc['conclusion'], conclusion=conc['conclusion'])
                    
                    # Save entailment probability
                    entailment_prob = answer_probs["Entailment"]
                    entailment_probs.append(entailment_prob)

                    # Save non-contradiction probability
                    non_contradiction_prob = answer_probs["Entailment"] + answer_probs["Neutral"]
                    non_contradiction_probs.append(non_contradiction_prob)

                    # Update entailment dict with ID of outgoing and incoming node and probabilities
                    entailment_dicts.append({"id": edge_id, "from": prev_conc["id"], "to": conc["id"], "ent_prob": entailment_prob, "ncont_prob": non_contradiction_prob})
                    edge_id +=1

                # Update previous conclusion list and previous conclusion ID
                prev_concs.append(conc)
                prev_id = conc["id"]
            

            # Step Entailment
            # Get mean and min non-contradiction probability
            avg_non_cont_prob = sum(non_contradiction_probs) / len(non_contradiction_probs)
            print(f"Average Non-Contadiction Prob: {avg_non_cont_prob}") if self.verbose >= 2 else None
            min_non_cont_prob = min(non_contradiction_probs)
            print(f"Average Non-Contadiction Prob: {min_non_cont_prob}") if self.verbose >= 2 else None

            # Get mean and min entailment probability
            avg_ent_prob = sum(entailment_probs) / len(entailment_probs)
            print(f"Average Entailment Prob: {avg_ent_prob}") if self.verbose >= 2 else None
            min_ent_prob = min(entailment_probs)
            print(f"Average Entailment Prob: {min_ent_prob}") if self.verbose >= 2 else None

            # Save results
            se_result = {
                "id": rc_id,
                "entailment_probability (min)": min_ent_prob,
                "entailment_probability (mean)": avg_ent_prob,
                "non-contradiction_probability (min)": min_non_cont_prob,
                "non-contradiction_probability (mean)": avg_non_cont_prob,
            }

            # Chain Entailment
            print(f"\nEntailment Dict: {entailment_dicts}") if self.verbose >= 1 else None
            for d in entailment_dicts:
                print(f"{d['id']} - From: {d['from']} - To: {d['to']} - EntProb: {d['ent_prob']} - nContProb: {d['ncont_prob']}") if self.verbose >= 2 else None
            
            # Find all paths from a premise to the final conclusion and their average probability
            ent_chains = self.find_all_paths_with_avg_probabilities(entailment_dicts, score="ent_prob")
            non_cont_chains = self.find_all_paths_with_avg_probabilities(entailment_dicts, score="ncont_prob")
            print(f"\nEnt Chains: {ent_chains}") if self.verbose >= 2 else None
            print(f"\nNon Cont Chains: {non_cont_chains}") if self.verbose >= 2 else None

            # Get the probability of all paths
            ent_chain_probs_list = [p['average_probability'] for p in ent_chains] 
            non_cont_chain_probs_list = [p['average_probability'] for p in non_cont_chains]
            print(f"\nAll Chain Ent Probs: {ent_chain_probs_list}") if self.verbose >= 2 else None
            print(f"\nAll Chain Non Cont Probs: {non_cont_chain_probs_list}") if self.verbose >= 2 else None

            # Weighted Chain Entailment
            # Entailment probabilities are weighted by multiplying the aggreement probability of the outgoing node
            w_entailment_dicts = self.multiply_agreement_and_entailment(entailment_dicts, agreement_dicts)
            print(f"\nAgreement Dict: {agreement_dicts}") if self.verbose >= 2 else None
            print(f"\nWeighted Entailment Dict: {w_entailment_dicts}") if self.verbose >= 2 else None
            for d in w_entailment_dicts:
                print(f"From: {d['from']} - To: {d['to']} - wEntProb: {d['w_ent_prob']} - wnContProb: {d['w_ncont_prob']}") if self.verbose == 2 else None
            
            # Find all paths from a premise to the final conclusion and their average probability
            w_ent_chains = self.find_all_paths_with_avg_probabilities(w_entailment_dicts, score="w_ent_prob")
            w_non_cont_chains = self.find_all_paths_with_avg_probabilities(w_entailment_dicts, score="w_ncont_prob")
            print(f"\nweighted ENT Chains: {w_ent_chains}") if self.verbose >= 2 else None
            print(f"\nweighted Non Cont Chains: {w_non_cont_chains}") if self.verbose >= 2 else None
            
            # Get the probability of all paths
            w_ent_chain_probs_list = [p['average_probability'] for p in w_ent_chains]
            w_non_cont_chain_probs_list = [p['average_probability'] for p in w_non_cont_chains]
            print(f"\nAll Chain Weighted Ent Probs: {w_ent_chain_probs_list}") if self.verbose >= 2 else None
            print(f"\nAll Chain Weighted Non Cont Probs: {w_non_cont_chain_probs_list}") if self.verbose >= 2 else None

            # Save results
            ce_result = {
                "id": rc_id,
                "entailement-chain (max)": max(ent_chain_probs_list),
                "non-contradiction-chain (max)": max(non_cont_chain_probs_list),
                "weighted_entailment-chain (max)": max(w_ent_chain_probs_list),
                "weighted_non-contradiction-chain (max)": max(w_non_cont_chain_probs_list)
            }

            if self.return_full_scores:

                sg_result["conc_agreement_probs"] = conclusion_agreement_probs
                sg_result["prem_agreement_probs"] = premise_agreement_probs
                
                se_result["non_cont_probs"] = non_contradiction_probs
                se_result["ent_probs"] = entailment_probs
                
                ce_result["noncont_chain_probs"] = non_cont_chain_probs_list
                ce_result["ent_chain_probs"] = ent_chain_probs_list
                ce_result["weighted_noncont_chain_probs"] = w_non_cont_chain_probs_list
                ce_result["weighted_ent_chain_probs"] = w_ent_chain_probs_list

            sg_result_list.append(sg_result)
            se_result_list.append(se_result)
            ce_result_list.append(ce_result)

        # Compute means for all metrics and save results
        sg_result_dict = {
            "means": calculate_means(
                dict_list=sg_result_list,
                keys=[
                    "premise_agreement (min)", 
                    "conclusion_agreement (min)",
                    "premise_agreement (mean)", 
                    "conclusion_agreement (mean)",
                ]
            ),
            "results": sg_result_list
        }

        se_result_dict = {
            "means": calculate_means(
                dict_list=se_result_list,
                keys=[
                    "entailment_probability (min)",
                    "entailment_probability (mean)",
                    "non-contradiction_probability (min)",
                    "non-contradiction_probability (mean)",
                ]
            ),
            "results": se_result_list
        }

        ce_result_dict = {
            "means": calculate_means(
                dict_list=ce_result_list,
                keys=[
                    "weighted_entailment-chain (max)",
                    "weighted_non-contradiction-chain (max)",
                    "entailement-chain (max)",
                    "non-contradiction-chain (max)"
                ]
            ),
            "results": ce_result_list
        }

        # Return results
        return sg_result_dict, se_result_dict, ce_result_dict


    def get_entailment_prob(self, premise, conclusion):
        
        # NLI-prompt with CoT examples
        instruction = """Given a premise and a hypothesis, determine if the hypothesis follows logically from the premise.
        Choose one of the following options:
        - Entailment: if the hypothesis is directly supported by the premise.
        - Neutral: if the hypothesis is neither clearly supported nor contradicted by the premise.
        - Contradiction: if the hypothesis is contradicted by the premise.
        """
        instruction_question = "Does the hypothesis follow logically from the premise? (Entailment/Neutral/Contradiction)"

        input_text = f"{instruction}\nPremise: {premise}\nHypothesis: {conclusion}\n{instruction_question}"
        print(f"\nInput Text: {input_text}") if self.verbose >= 2 else None

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

        input_msg = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )  
        
        # Generate response to check
        if self.verbose >= 2:
            generation_args = {
                "max_new_tokens": 5,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }

            output = self.pipeline(input_msg, **generation_args)
            text = output[0]['generated_text']
            print(f"Generated Text: {text}\n")
        
        # Define answer choices
        answers = ["Entailment", "Neutral", "Contradiction"]

        # Tokenize the NLI prompt
        prompt_inputs = self.tokenizer(input_msg, return_tensors="pt")
        prompt_ids = prompt_inputs.input_ids

        # Dictionary to store probabilities for each answer
        answer_probs = {}

        for answer in answers:

            # Tokenize the answer independently
            answer_inputs = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt")
            answer_ids = answer_inputs.input_ids

            # Concatenate prompt and answer ids
            input_ids = torch.cat([prompt_ids, answer_ids], dim=-1)

            # Store inverse perplexity for each answer
            answer_probs[answer] = self.get_inv_perplexity(input_ids, answer_ids)

        # Normalize to get probabilities for each answer
        total_prob = sum(answer_probs.values())
        for answer in answers:
            answer_probs[answer] /= total_prob

        # Print probabilities
        for answer, prob in answer_probs.items():
            print(f"Probability of '{answer}': {prob}") if self.verbose >= 2 else None

        return answer_probs

    
    def get_inv_perplexity(self, input_ids, answer_ids):
        
        # Compute log probabilities with the model
        with torch.no_grad():
            outputs = self.model(input_ids)
            log_probs = F.log_softmax(outputs.logits, dim=-1)

        # Sum log probabilities
        answer_log_prob = sum(log_probs[0, -(1 + len(answer_ids[0])) + i, token_id].item() for i, token_id in enumerate(answer_ids[0]))

        # Adjust for token length
        average_log_prob = answer_log_prob / len(answer_ids[0])

        # Compute perplexity
        perplexity = torch.exp(-torch.tensor(average_log_prob))  # Adjusted perplexity

        # Store adjusted probability for each answer
        return 1 / perplexity.item()
    

    def find_all_paths_with_avg_probabilities(self, edges, score):
        
        # Build the adjacency list
        adj_list = defaultdict(list)
        nodes = set()

        for edge in edges:
            outgoing = edge['from']
            incoming = edge['to']
            probability = edge[score]
            
            adj_list[outgoing].append((incoming, probability))
            nodes.add(outgoing)
            nodes.add(incoming)

        # Find all starting nodes (with zero incoming edges)
        indegree = defaultdict(int)
        for edge in edges:
            incoming = edge['to']
            indegree[incoming] += 1
        start_nodes = [node for node in nodes if indegree[node] == 0]

        # Recursively explore all paths
        def explore_paths(node, path, total_prob):
            path = path + [node]
            
            # Compute average probability if the final node is reached
            if not adj_list[node]:
                avg_prob = total_prob / (len(path)-1) if path else 0
                all_paths.append({'path': path, 'average_probability': avg_prob})
                return
            
            # Otherwise continue exploring
            for neighbor, prob in adj_list[node]:
                explore_paths(neighbor, path, total_prob + prob)

        all_paths = []

        # Explore paths starting from each start node
        for start_node in start_nodes:
            explore_paths(start_node, [], 0)
        print(f"All found Paths {all_paths}") if self.verbose >= 2 else None
        return all_paths


    def multiply_agreement_and_entailment(self, entailment_dicts, agreement_dicts):
        
        # Crate lookup dict for agreement probs
        agreement_dict = {d["node"]: d["agreement_prob"] for d in agreement_dicts}

        for d1 in entailment_dicts:
            if d1["from"] in agreement_dict:
                # Multiply probabilities of entailment dict with the agreement probabilities
                d1["w_ncont_prob"] = d1["ncont_prob"] * agreement_dict[d1["from"]]
                d1["w_ent_prob"] = d1["ent_prob"] * agreement_dict[d1["from"]]

        # Return weighted dict
        return entailment_dicts


if False:
    def find_highest_average_path(edges, score):
        # Step 1: Build the adjacency list
        adj_list = defaultdict(list)
        indegree = defaultdict(int)
        nodes = set()

        for edge in edges:
            outgoing = edge['from']
            incoming = edge['to']
            probability = edge[score]
            
            adj_list[outgoing].append((incoming, probability))
            indegree[incoming] += 1
            nodes.add(outgoing)
            nodes.add(incoming)

        # Step 2: Find all starting nodes (with zero incoming edges)
        start_nodes = [node for node in nodes if indegree[node] == 0]

        # Step 3: Initialize tracking dictionaries
        max_avg_prob = defaultdict(lambda: float('-inf'))
        total_prob = defaultdict(lambda: 0)
        path_count = defaultdict(lambda: 0)
        predecessors = {}
        
        # Step 4: Process nodes in topological order
        queue = deque(start_nodes)
        for start in start_nodes:
            total_prob[start] = 0
            path_count[start] = 0
            max_avg_prob[start] = 0
        
        while queue:
            node = queue.popleft()
            
            for neighbor, prob in adj_list[node]:
                # Update cumulative probability and path count for the neighbor
                total_prob[neighbor] = total_prob[node] + prob
                path_count[neighbor] = path_count[node] + 1
                avg_prob = total_prob[neighbor] / path_count[neighbor]
                
                # Update if this path to the neighbor has a higher average probability
                if avg_prob > max_avg_prob[neighbor]:
                    max_avg_prob[neighbor] = avg_prob
                    predecessors[neighbor] = node
                
                # Decrease indegree and add to queue if all prerequisites are done
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Step 5: Identify the final node (the one with no outgoing edges)
        print(nodes)
        print(adj_list)
        final_node = None
        for node in nodes:
            if not adj_list[node]:
                final_node = node
                break

        print(final_node)

        # Step 6: Reconstruct the path to the final node with the highest average probability
        if final_node is None or max_avg_prob[final_node] == float('-inf'):
            return "No valid path to the final node found"
        
        path = []
        node = final_node
        while node in predecessors:
            path.append(node)
            node = predecessors[node]
        path.append(node)  # Add the starting node

        path.reverse()  # Reverse to get the path from start to final

        return {
            'path': path,
            'average_probability': max_avg_prob[final_node]
        }


if False:
    def find_all_paths_to_node(edges, final_node):
        # Step 1: Build an adjacency list with reversed edges
        reverse_adj_list = defaultdict(list)
        for edge in edges:
            outgoing = edge['from']
            incoming = edge['to']
            reverse_adj_list[incoming].append(outgoing)

        # Step 2: Recursively explore all paths to the final node
        def explore_paths(node, path):
            path = path + [node]
            # If there's no predecessor for the current node, we've reached the start of a path
            if not reverse_adj_list[node]:
                all_paths.append(path[::-1])  # Reverse the path to get it from start to end
                return
            
            for predecessor in reverse_adj_list[node]:
                explore_paths(predecessor, path)

        all_paths = []
        explore_paths(final_node, [])

        return all_paths





