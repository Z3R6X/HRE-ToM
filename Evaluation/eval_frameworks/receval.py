from tqdm import tqdm
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from allennlp.predictors.predictor import Predictor
import re, string
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize, sent_tokenize
import nltk


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util')))
from util.eval import split_all_chains, calculate_means


# Most of the functions in this file were taken from:
# https://github.com/archiki/ReCEval/blob/main/evaluate_receval.py
# with only small modifications
class ReCEval:
    def __init__(
            self,
            cache_dir=None,
            device="cuda",
            return_full_scores=False,
        ):

        self.name = "receval"
        self.device = device
        self.return_full_scores = return_full_scores

        self.K = [0, 4, 8]

        self.srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

        ent_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        self.ent_tokenizer = AutoTokenizer.from_pretrained(ent_model_name)
        self.ent_model = AutoModelForSequenceClassification.from_pretrained(
            ent_model_name,
            cache_dir=cache_dir
        ).to(self.device)

    
    ### Preprocessing
    def load_predictions(self, predictions):

        # Load source texts and reasoning chains
        print("\nLoad Sources and Predictions")
        sources = [d['scenario'] for d in predictions]
        reasoning_chains = [d['prediction'] for d in predictions]

        self.num_responses = len(reasoning_chains)

        # Split source texts
        self.sources = sources
        self.split_sources = split_all_chains(self.sources)
        
        # Split reasoning chains
        self.chains = reasoning_chains
        self.split_chains = split_all_chains(self.chains)

        # Extract RCUs
        print("\nExtract RCUs")
        self.rcu_chains = self.get_all_rcus(self.split_chains)


    def get_all_rcus(self, split_chains):
        return [self.get_rcus(split_chain) for split_chain in tqdm(split_chains, total=len(split_chains))]
    

    def get_rcus(self, split_chain):
        
        rcu_chain = []

        # Extract RCUs from each step of the reasoning chain
        for split in split_chain:
            units = get_phrases(
                srl_predictor=self.srl_predictor,
                sent=split
            )
            premise_units, conc_units = [], []
            premise_units.extend(units[:-1])
            conc_units.append(units[-1])

            rcu_chain.append(
                {
                    "premises": premise_units,
                    "conclusion": conc_units
                }
            )

        return rcu_chain
    

    ### Evaluation
    def evaluate_all(self):

        # Get results for:
        # Intra-Correctness and
        # Inter-Correctness

        result_list = list()

        for id, (rcu_chain, split_source) in tqdm(enumerate(zip(self.rcu_chains, self.split_sources)), total=self.num_responses):
            
            result = {
                "id": id
            }

            # Evaluate Intra-Correctness
            for k in self.K:
                ues = self.evaluate_unit_entailment_score(rcu_chain, k)

                result[f"intra-correctness (min) (k={k})"] = ues[0]
                result[f"intra-correctness (mean) (k={k})"] = ues[1]

                if self.return_full_scores:
                    result[f"intra-correctness_scores (k={k})"] = ues[2]

            # Evaluate Inter-Correctness
            ucs = self.evaluate_unit_contradict_score(rcu_chain, input_context_sentences=split_source)
        
            result["inter-correctness (min)"] = ucs[0]
            result["inter-correctness (mean)"] = ucs[1]
            
            if self.return_full_scores:
                result["inter-correctness_scores"] = ucs[2]
            
            result_list.append(result)

        intra_keys = [f"intra-correctness (min) (k={k})" for k in self.K]
        intra_mean_keys = [f"intra-correctness (mean) (k={k})" for k in self.K]

        # Compute means for all metrics and save results
        result_dict = {
            "K": self.K,
            "means": calculate_means(
                dict_list=result_list,
                keys=[
                    "inter-correctness (min)",
                    "inter-correctness (mean)",
                    ] + intra_keys + intra_mean_keys
                ),
            "results": result_list
        }

        result_dict = {
            "rcu": result_dict
        }

        return result_dict


    def evaluate_unit_entailment_score(self, rcu_chain, K=0):
        
        alt_step_ent_scores = []
        running_conc = []

        # Get entailment probabilities from premises to conclusion within one step
        for step in rcu_chain: 
            premise_units = step["premises"]
            conc_units = step["conclusion"]

            alt_step_ent_scores.append(self.obtain_unit_entailment_score(premise_units + running_conc[-1*K:], conc_units))
            running_conc.extend(conc_units)

        # Get mean and min entailment probability
        min_ent_score = min(alt_step_ent_scores)
        mean_ent_score = sum(alt_step_ent_scores) / len(alt_step_ent_scores)
        return [min_ent_score, mean_ent_score, alt_step_ent_scores]

    
    def evaluate_unit_contradict_score(self, rcu_chain, input_context_sentences):
        
        step_contradict_scores = []
        running_conc = []

        # Get contradiction probabilities from conclusion to conclusion between steps
        for step in rcu_chain:
            conc_units = step["conclusion"]

            step_contradict_scores.append(self.obtain_contradiction_score(input_context_sentences + running_conc, conc_units))
            running_conc.extend(conc_units)

        # Get mean and min contradiction probability
        min_cont_score = min(step_contradict_scores)
        mean_cont_score = sum(step_contradict_scores) / len(step_contradict_scores)
        return [min_cont_score, mean_cont_score, step_contradict_scores]


    def obtain_entailment_scores(self, premise, hypothesis):
        input = self.ent_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            
            output = self.ent_model(input["input_ids"].to(self.device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
        return prediction['entailment']


    def obtain_contradiction_scores(self, premise, hypothesis):
        input = self.ent_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.ent_model(input["input_ids"].to(self.device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
        return prediction['contradiction']


    def obtain_unit_entailment_score(self, prem_units, conc_units):
        if len(prem_units):
            premise = ' and '.join(prem_units)
            hypothesis = ' and '.join(conc_units)
            score = self.obtain_entailment_scores(premise, hypothesis)
        else:
            score = 1
        return score


    def obtain_contradiction_score(self, prem_units, conc_units):
        pair_scores = []
        hypothesis = ' and '.join(conc_units)
        for premise in prem_units:
            pair_scores.append(self.obtain_contradiction_scores(premise, hypothesis))
        if len(pair_scores):
            score = 1 - max(pair_scores)
        else:
            score = 1
        
        return score


    def get_reasoning_chain_text(steps, sentences):
        # If using the reasoning trees directly
        step_texts = []
        covered_nodes = []
        for step in steps:
            parent_text = " and ".join([sentences[p] for p in step['parents'] if p not in covered_nodes])
            if len(parent_text): step_text = parent_text + ', so ' + sentences[step['child']] + "."
            else: step_text =  'so ' + sentences[step['child']] + '.' 
            covered_nodes.extend(step['parents']); covered_nodes.append(step['child'])
            step_texts.append(step_text)
        return step_texts


def get_phrases(srl_predictor, sent):
    # Simple RCU extractor without conjunction check for premises
    phrases = []
    history = ''
    srl_out = srl_predictor.predict(sent) 
    words = srl_out['words']  
    frames = [s['tags'] for s in srl_out['verbs']]
    descs = [s['description'] for s in srl_out['verbs']]
    mod_sent = detokenize(words).rstrip('.')
    for frame, desc in zip(frames, descs):
        phrase = extract_frame(frame, words, desc)
        if phrase == mod_sent: phrase = remove_modifiers(phrase, verb_modifiers(desc))
        phrases.append(phrase)
    phrases.sort(key=lambda s: len(s), reverse=True)
    filtered_phrases = []
    for p in phrases: 
        if p not in history:  
            history += ' ' + p
            filtered_phrases.append(p)
    if len(filtered_phrases): 
        filtered_phrases.sort(key=lambda s: mod_sent.find(s))
        left = mod_sent
        mod_filt = False
        for fp in filtered_phrases: left = left.replace(fp, '#').strip(string.punctuation + ' ')
        for l in left.split('#'): 
            l = l.strip(string.punctuation + ' ')
            if len(l.split()) >=4 and l not in " ".join(filtered_phrases): 
                verb_match = ['VB' in k[1] for k in nltk.pos_tag(word_tokenize(l))]
                if sum(verb_match):
                    filtered_phrases.append(l)
                    mod_filt = True
        if mod_filt: filtered_phrases.sort(key=lambda s: mod_sent.find(s))
        return filtered_phrases
    else: return [sent.rstrip('.')]


def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


def extract_frame(tags, words, desc):
    prev = 'O'
    start, end = None, None
    if len(set(tags)) == 1: return ''
    tags = [t if 'C-ARG' not in t else 'O' for t in tags] #check if the modifier is a verb phrase
    for w in range(len(words)):
        if 'B-' in tags[w] and start is None: start = w
        if tags[len(words) - w -1]!='O' and end is None: end = len(words) - w -1 
    
    if end is None: end = start
    sent = detokenize(words[start: end + 1]).rstrip('.')
    return sent


def remove_modifiers(sent, modifiers):
    if not len(modifiers): return sent
    for mod in modifiers:
        sent = sent.replace(mod, "")
        sent = re.sub(' +', ' ', sent) # remove any double spaces
        sent = sent.strip(string.punctuation + ' ') # remove stray punctuations
    return sent


def verb_modifiers(desc):
    filtered_mods = []
    mods = re.findall(r"\[ARGM.*?\]", desc)
    if not len(mods): return filtered_mods
    for mod in mods:
        phrase = mod.split(': ')[1].rstrip(']')
        verb_match = ['VB' in k[1] for k in nltk.pos_tag(word_tokenize(phrase))]
        if sum(verb_match) and len(phrase.split()) > 2: filtered_mods.append(phrase) # put in a length criteria
    return filtered_mods


def get_sent_phrases(para):
    sentences = sent_tokenize(para)
    phrases = []
    for sent in sentences:
        phrases.extend(get_phrases(sent))
    return phrases
    