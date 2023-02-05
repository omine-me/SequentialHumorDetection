import copy
import csv
import os

model_name = "roberta-large"
gen_model_name = "facebook/opt-125m"
out_file = r'C:\Users\omine\Documents\M1_r\HugTF\main\state_'+model_name.replace("/", "_")+'_sm_topP_multiDistance_BERT_lasthidden.csv'

try:
    with open(out_file, "w", newline='') as f:
        pass
except PermissionError:
    print("File not closed")

PRED_WORD_COUNT = 1
DISTANCE_WORDS_COUNT = 3
TOP_P = 0.9

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
# from accelerate import Accelerator
import torch
from torch.nn import functional as F
import numpy as np

tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

gen_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
gen_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# bertscore = load("bertscore")

prompt = "Lol well the fans in Philli are infamous so I'm not surprised there was an actual court inside the stadium"

def count_top_p(scores, top_p, return_top_n=10):
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)  #大きい順
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    print("top_n:", tokenizer.batch_decode(sorted_indices[0][:return_top_n], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    
    return np.count_nonzero(cumulative_probs < top_p), sorted_indices[0][:return_top_n]


with open(out_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["", "distance", "distance_from_prev"])

    splitted = gen_tokenizer(prompt, return_tensors="pt")
    for i in range(len(splitted.input_ids[0])):
        if i == 0:
            print("0")
            continue

        generate_ids = gen_model.generate((splitted.input_ids[0][:i])[np.newaxis, :].cuda(), max_new_tokens=PRED_WORD_COUNT,
                                           output_scores=True, return_dict_in_generate=True)
        
        inputs = tokenizer(gen_tokenizer.batch_decode((splitted.input_ids[0][:i+PRED_WORD_COUNT])[np.newaxis, :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                                , return_tensors="pt")
        outputs = model(**inputs)
        prev_inputs = tokenizer(gen_tokenizer.batch_decode((splitted.input_ids[0][:i])[np.newaxis, :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                                , return_tensors="pt")
        prev_outputs = model(**prev_inputs)

        gen_inputs = tokenizer(tokenizer.batch_decode(generate_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], return_tensors="pt")
        gen_outputs = model(**gen_inputs)


        distance = np.mean(np.abs(outputs.last_hidden_state.detach().numpy()[:,-1:,:] 
                                    - gen_outputs.last_hidden_state.detach().numpy()[:,-1:,:]))
        
        distance_from_prev = np.mean(np.abs(outputs.last_hidden_state.detach().numpy()[:,-1:,:] 
                                    - prev_outputs.last_hidden_state.detach().numpy()[:,-1:,:]))

        writer.writerow([tokenizer.decode(splitted.input_ids[0][i]), distance, distance_from_prev])