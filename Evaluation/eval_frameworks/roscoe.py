from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util')))
from util.eval import split_all_chains, calculate_means


class Roscoe:
    def __init__(
            self,
            cache_dir="/data/hleier/MA/EvalModels",
            device="cuda",
            return_full_scores=False,
            #verbose=0
        ):
        
        self.name = "roscoe"
        self.return_full_scores = return_full_scores
        #self.verbose = verbose

        # Embedding
        self.embedding_model = SentenceTransformer(
            "facebook/roscoe-512-roberta-base",
            cache_folder = cache_dir,
            device = device
        )  

        # NLI
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            cache_dir = cache_dir,
        )
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            cache_dir = cache_dir,
        )

        # https://huggingface.co/docs/transformers/perplexity
        self.perplexity_tokenizer = GPT2TokenizerFast.from_pretrained(
            "openai-community/gpt2-large",
            cache_dir = cache_dir,
        )
        self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token
        self.perplexity_model = GPT2LMHeadModel.from_pretrained(
            "openai-community/gpt2-large",
            cache_dir = cache_dir,
        )
       

    def load_predictions(self, predictions):
        
        # Load source texts and reasoning chains
        print("\nLoad Sources and Predictions")
        sources = [d['scenario'] for d in predictions]
        reasoning_chains = [d['prediction'] for d in predictions]

        self.reasoning_chains = reasoning_chains
        self.num_responses = len(reasoning_chains)

        # Embed full chains 
        self.sources_chain_emb = self.embed_all_chains(sources)
        self.full_reasoning_chains_emb = self.embed_all_chains(reasoning_chains)
        
        # Split source texts and reasoning chains into steps
        self.sources_split = split_all_chains(sources)
        self.reasoning_chains_split = split_all_chains(reasoning_chains)

        # Embed steps
        self.sources_emb = self.embed_all_splits(self.sources_split)
        self.reasoning_chains_emb = self.embed_all_splits(self.reasoning_chains_split)


    def embed_splits(self, split_chain):
        embedding_list = self.embedding_model.encode(split_chain, convert_to_numpy=False)
        return embedding_list


    def embed_all_splits(self, split_chains):
        embedding_lists = [self.embed_splits(split_chain) for split_chain in split_chains]
        return embedding_lists


    def embed_chain(self, chain):
        embedding = chain
        return embedding
    

    def embed_all_chains(self, chains):
        embedding_list = self.embedding_model.encode(chains, convert_to_numpy=False)
        return embedding_list


    ### Evaluation
    def evaluate_all(self):
        # Get results for:
        # Semantic Alignment,
        # Semantic Similarity,
        # Logical Inference and
        # Language Coherence
        print("\nEvaluate Semantic Alignment")
        sa = self.evaluate_semantic_alignment()
        print("\nEvaluate Semantic Similartiy")
        ss = self.evaluate_semantic_similarity()
        print("\nEvaluate Logical Inference")
        li = self.evaluate_logical_inference()
        print("\nEvaluate Language Coherence")
        lc = self.evaluate_language_coherence()


        full_dict = {
            "semantic_alignment": sa,
            "semantic_similarity": ss,
            "logical_inference": li,
            "language_coherence": lc,
        }

        return full_dict


    ### Semantic Alignment Metrics
    def evaluate_semantic_alignment(self):

        result_list = []  

        for id, (reasoning_step_emb, src_emb) in tqdm(enumerate(zip(self.reasoning_chains_emb, self.sources_emb)), total=self.num_responses):
            fss = self.faithfulness_step_score(reasoning_step_emb, src_emb)
            iss = self.informativeness_step_score(reasoning_step_emb, src_emb)
            result = {
                "id": id,
                "faithfulness_step": fss.item(),
                "informativeness_step": iss.item()
            }
            result_list.append(result)

        # Compute means for all metrics and save results
        result_dict = {
            "means": calculate_means(
                dict_list=result_list,
                 keys=[
                    "faithfulness_step",
                    "informativeness_step"
                    ]
                ),
            "results": result_list
        }

        return result_dict


    def r_align(self, candidate_emb, reference_emb):   
        scores = []
        # Get alignment score for each candidate step
        for i, c in enumerate(candidate_emb):
            # Get similarity to each referene step
            similarities = [F.cosine_similarity(c.unsqueeze(0), r.unsqueeze(0)) for r in reference_emb]
            # Save highest similarity
            scores.append((1 + torch.max(torch.stack(similarities))) / 2)

        # Return alignment vector
        return torch.stack(scores)


    def faithfulness_step_score(self, reasoning_step_emb, src_emb):
        # Get mean alignment vector
        r_align_vector = self.r_align(candidate_emb=reasoning_step_emb, reference_emb=src_emb)
        score = torch.mean(r_align_vector)
        
        return score 


    def informativeness_step_score(self, reasoning_step_emb, src_emb):
        # Get mean alignment vector for source steps and reasoning steps
        reasoning_chain_to_ref = torch.mean(self.r_align(candidate_emb=reasoning_step_emb, reference_emb=src_emb))
        ref_to_reasoning_chain = torch.mean(self.r_align(candidate_emb=src_emb, reference_emb=reasoning_step_emb))

        score = (reasoning_chain_to_ref + ref_to_reasoning_chain) / 2
        
        return score


    ### Semantic Similarity Metrics
    def evaluate_semantic_similarity(self):

        result_list = list()  

        for id, (full_reasoning_chain_emb, src_chain_emb, reasoning_step_emb) in tqdm(enumerate(zip(self.full_reasoning_chains_emb,
         self.sources_chain_emb, self.reasoning_chains_emb)), total=self.num_responses):
            ics = self.info_chain_score(full_reasoning_chain_emb, src_chain_emb)
            rss = self.repetition_step_score(reasoning_step_emb)
            result = {
                "id": id,
                "info_chain": ics.item(),
                "repetition_step": rss.item()
            }
            result_list.append(result)

        # Compute means for all metrics and save results
        result_dict = {
            "means": calculate_means(
                dict_list=result_list,
                keys=[
                    "info_chain",
                    "repetition_step"
                ]
            ),
            "results": result_list
        }

        return result_dict
    

    def info_chain_score(self, full_reasoning_chain_emb, full_src_emb):
        # Get similarity between full reasoning chain and source text
        return (1 + F.cosine_similarity(full_reasoning_chain_emb.unsqueeze(0), full_src_emb.unsqueeze(0))) / 2


    def repetition_step_score(self, reasoning_step_emb): 
        scores = []
        # Get similarity to each previous reasoning step
        for i in range(1, len(reasoning_step_emb)):
            for rs_prev in reasoning_step_emb[:i]:
                scores.append(F.cosine_similarity(reasoning_step_emb[i].unsqueeze(0), rs_prev.unsqueeze(0)))
        if not scores:
            scores.append(torch.Tensor([1]))
        return (1 - torch.max(torch.stack(scores))) / 2


    def evaluate_logical_inference(self):

        result_list = list()  

        for id, (reasoning_chain_split, src_split) in tqdm(enumerate(zip(self.reasoning_chains_split, self.sources_split)), total=self.num_responses):
            selfcs = self.self_consistency_score(reasoning_chain_split)
            sourcecs = self.source_consistency_score(reasoning_chain_split, src_split)
            result = {
                "id": id,
                "self_consistency (min)": selfcs[0].item(),
                "source_consistency (min)": sourcecs[0].item(),
                "self_consistency (mean)": selfcs[1].item(),
                "source_consistency (mean)": sourcecs[1].item(),
            }
            result_list.append(result)

            if self.return_full_scores:
                result["self_consistency_scores"] = selfcs[2]
                result["source_consistency_scores"] = sourcecs[2]

        # Compute means for all metrics and save results
        result_dict = {
            "means": calculate_means(
                dict_list=result_list,
                keys=[
                    "self_consistency (min)",
                    "source_consistency (min)",
                    "self_consistency (mean)",
                    "source_consistency (mean)"
                ]
            ),
            "results": result_list
        }

        return result_dict


    def self_consistency_score(self, reasoning_chain):
        
        scores = []

        # Get entailment from all previous steps to current step
        for i in range(1, len(reasoning_chain)):
            
            for rs_prev in reasoning_chain[:i]:
                
                # Tokenize input
                input = self.nli_tokenizer(
                    rs_prev, reasoning_chain[i], truncation=True, padding=True, return_tensors="pt"
                )
                
                # Get prediction
                with torch.no_grad():
                    output = self.nli_model(**input)
                prediction = torch.softmax(output["logits"], -1).tolist()
                
                # Save prediction
                scores.append(prediction[0][2])

        if not scores:
            scores.append(torch.Tensor([1]))
        return [1 - torch.max(torch.Tensor(scores)), 1-torch.mean(torch.Tensor(scores)), [1-s for s in scores]]

    
    def source_consistency_score(self, reasoning_chain, src):
        
        scores = []
        
        # Get entailment from each source step to each reasoning step
        for rs in reasoning_chain:
            for s in src:

                # Tokenize input
                input = self.nli_tokenizer(
                    rs, s, truncation=True, padding=True, return_tensors="pt"
                )

                # Get prediction
                with torch.no_grad():
                    output = self.nli_model(**input)
                prediction = torch.softmax(output["logits"], -1).tolist()

                # Save prediction
                scores.append(prediction[0][2])

        if not scores:
            scores.append(torch.Tensor([1]))
        return [1 - torch.max(torch.Tensor(scores)), 1-torch.mean(torch.Tensor(scores)), [1-s for s in scores]]


    ### Language Coherence Metrics
    def evaluate_language_coherence(self):

        result_list = list()  

        for id, (reasoning_chain_chain, reasoning_chain_split) in tqdm(enumerate(zip(self.reasoning_chains, self.reasoning_chains_split)), total=self.num_responses):
            pcs = self.perplexity_chain_score(reasoning_chain_chain)
            pss = self.perplexity_step_score(reasoning_chain_split)
            result = {
                "id": id,
                "perplexity_chain": pcs.item(),
                "perplexity_step": pss.item()
            }
            result_list.append(result)

        # Compute means for all metrics and save results
        result_dict = {
            "means": calculate_means(
                dict_list=result_list,
                keys=[
                    "perplexity_chain",
                    "perplexity_step"
                    ]
                ),
            "results": result_list
        }

        return result_dict

    def perplexity_chain_score(self, reasoning_chain):
        
        # Tokenize reasoning chain
        input = self.perplexity_tokenizer(reasoning_chain, truncation=True, padding=True, return_tensors="pt")
        target_ids = input.input_ids

        # Get the model perplexity
        with torch.no_grad():
            output = self.perplexity_model(input.input_ids, labels=target_ids)
            nll = output.loss
        perplexity = torch.exp(torch.mean(nll))
        
        # Return inverse perplexity
        return 1 / perplexity


    def perplexity_step_score(self, reasoning_chain_split):
        
        perplexities = []

        for rs in reasoning_chain_split: 

            # Tokenize reasoning step 
            input = self.perplexity_tokenizer(rs, truncation=True, padding=True, return_tensors="pt")
            target_ids = input.input_ids

            # Get the model perplexity
            with torch.no_grad():
                output = self.perplexity_model(input.input_ids, labels=target_ids)
                nll = output.loss
            perplexity = torch.exp(torch.mean(nll))
            perplexities.append(perplexity)

        # Get mean perplexity
        mean_perplexity = torch.mean(torch.stack(perplexities))
        
        # Return inverse mean perplexity
        return 1 / mean_perplexity

