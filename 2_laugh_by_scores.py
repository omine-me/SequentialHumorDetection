import copy
import csv
import os

model_name = "facebook/opt-125m"
# model_name = "facebook/opt-6.7b"
out_file = r'C:\Users\omine\Documents\M1_r\HugTF\main\state_'+model_name.replace("/", "_")+'_scores.csv'

try:
    with open(out_file, "w", newline='') as f:
        pass
except PermissionError:
    print("File not closed")

# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import OPTModel, AutoTokenizer, AutoModelForCausalLM
# from accelerate import Accelerator
import torch
from torch.nn import functional as F
import numpy as np

# model = OPTModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
gen_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
# accelerator = Accelerator()
# model = accelerator.prepare(model)

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# prompt = "Why they didn't say that in the air publicly distributed. That was weird. I didn't read that. Okay, let's not talk about terrorism. Now, Janice was talking to one of her friends and she said, oh, you know, if you're going to get pregnant, you have to stop smoking dope. That's not good. You see, we rented that one of those Tom Clanty movies with Harrison Ford, the one where he goes to Columbia. Did you see that? No, no."
# prompt = "Hi. How is the weather today? It's sunny today. So I will stay home all day."
# prompt = "My two-year-old son is a president of the U.S."
# prompt = "My dream is to be a president of the U.S."
# prompt = "The kitchen stinks. I'll throw out the garbage. So Dick, how about getting some coffee for tonight? Coffee? I don't honestly like that kind of stuff. Come on, you can at least try a little, besides your cigarette. What's wrong with that? Cigarette is the thing I go crazy for. Not for me, Dick. Are things still going badly with your houseguest? Getting worse. Now he's eating me out of house and home. I'Ve tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. You're right. Everything is probably going to come to a head tonight. I'll keep you informed. "
prompt = "Lol well the fans in Philli do have a bad reputation so I'm not surprised there was an actual court inside the stadium"
# prompt = "Lol well the fans in Philli are infamous so I'm not surprised there was an actual court inside the stadium"
# prompt = "Tokyo is a capital of Japan. It's common sense."


def get_laughness(scores, current_word_id):
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)  #大きい順
    sorted_logits = F.softmax(sorted_logits, dim=-1)
    # get index
    rank = ((sorted_indices[0] == current_word_id).nonzero(as_tuple=True)[0]).item()
    score = sorted_logits[0][rank].item()
    
    return rank*score, rank, score


with open(out_file, "w", newline='') as f:
    writer = csv.writer(f)

    splitted = tokenizer(prompt, return_tensors="pt")
    for i in range(len(splitted.input_ids[0])):
        if i == 0:
            print("0")
            continue
        generate_ids = gen_model.generate((splitted.input_ids[0][:i])[np.newaxis, :].cuda(), max_new_tokens=1,
                                           output_scores=True, return_dict_in_generate=True)
                                           
        laughness, rank, score = get_laughness(generate_ids.scores[0].cpu().to(torch.float32), splitted.input_ids[0][i].item())

        writer.writerow([tokenizer.decode(splitted.input_ids[0][i]), laughness, rank, score])
