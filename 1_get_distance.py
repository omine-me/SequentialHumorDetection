import copy
import csv
import os

model_name = "facebook/opt-125m"
# model_name = "facebook/opt-6.7b"
out_file = r'C:\Users\omine\Documents\M1_r\HugTF\main\state_'+model_name.replace("/", "_")+'_sm_topP_multiDistance_long.csv'

try:
    with open(out_file, "w", newline='') as f:
        pass
except PermissionError:
    print("File not closed")

DISTANCE_WORDS_COUNT = 10
TOP_P = 0.9

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
prompt = "Lol well the fans in Philli are infamous so I'm not surprised there was an actual court inside the stadium"
# prompt = "Tokyo is a capital of Japan. It's common sense."


def count_top_p(scores, top_p, return_top_n=10):
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)  #大きい順
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    print("top_n:", tokenizer.batch_decode(sorted_indices[0][:return_top_n], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    
    return np.count_nonzero(cumulative_probs < top_p), sorted_indices[0][:return_top_n]


# prev_state = None
# splitted = prompt.split(" ")
# for i in range(len(splitted)):
#     input = tokenizer(" ".join(splitted[:i+1]), return_tensors="pt")
#     out = model(**input)
#     """
#     last_hidden_state.shape:
#     Batch Size：一度にモデルに処理される入力系列データの数（バッチの大きさ）
#     Sequence Length：一個の入力系列データの長さ（含まれるトークンの数） config: max_position_embeddings
#     Hidden Size：各トークンの意味を表現するためのベクトルの次元
#     """
#     curr_state = out.last_hidden_state.detach().numpy()[:,-1:,:]
#     if prev_state is not None:
#         distance = np.mean(np.abs(curr_state - prev_state))
#         print(distance)
#     prev_state = copy.deepcopy(curr_state)

with open(out_file, "w", newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(["", "Intensity"])

    splitted = tokenizer(prompt, return_tensors="pt")
    for i in range(len(splitted.input_ids[0])):
        if i == 0:
            print("0")
            continue
        generate_ids = gen_model.generate((splitted.input_ids[0][:i])[np.newaxis, :].cuda(), max_new_tokens=1,
                                           output_scores=True, return_dict_in_generate=True)
        generate_long_ids = gen_model.generate((splitted.input_ids[0][:i])[np.newaxis, :].cuda(), max_new_tokens=5,
                                           output_scores=True, return_dict_in_generate=True)
        
        ### 予測の確信度を図る（特定の単語になるほど高い）
        # smd = torch.nn.functional.softmax(generate_ids.scores[0].squeeze().cpu().to(torch.float32))
        # print(smd.min(), smd.max())
        # certainty = 1/np.count_nonzero((smd>=.05))
        print(tokenizer.decode(splitted.input_ids[0][i]))
        top_p_count, top_words = count_top_p(generate_ids.scores[0].cpu().to(torch.float32), TOP_P, DISTANCE_WORDS_COUNT)
        certainty = 1/top_p_count
        
        # generated_sentance = tokenizer.batch_decode(generate_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # gen_sentence = tokenizer(generated_sentance, return_tensors="pt")

        # gt_sentence = {'input_ids': (splitted.input_ids[0][:i+1])[np.newaxis, :],'attention_mask': (splitted.attention_mask[0][:i+1])[np.newaxis, :]}
        gt_sentence = {'input_ids': (splitted.input_ids[0][:i+1])[np.newaxis, :]}
        gt_out = gen_model(**gt_sentence, output_hidden_states=True)
        gt_state = gt_out.hidden_states[-1].detach().numpy()[:,-1:,:]
        
        distances = []
        for word_id in top_words:
            generate_ids.sequences[0][-1] = word_id
            # gen_out = gen_model(**gen_sentence, output_hidden_states=True)
            gen_out = gen_model(generate_ids.sequences, output_hidden_states=True)
            gen_state = gen_out.hidden_states[-1].detach().cpu().numpy()[:,-1:,:]
            distances.append(np.mean(np.abs(gt_state - gen_state)))

        distance = min(distances)

        # try:
        gt_long_sentence = {'input_ids': (splitted.input_ids[0][:i+5])[np.newaxis, :]}
        gt_long_out = gen_model(**gt_long_sentence, output_hidden_states=True)
        gt_long_state = gt_long_out.hidden_states[-1].detach().cpu().numpy()[:,-1:,:]

        gen_long_out = gen_model(generate_long_ids.sequences, output_hidden_states=True)
        gen_long_state = gen_long_out.hidden_states[-1].detach().cpu().numpy()[:,-1:,:]

        long_distance = np.mean(np.abs(gt_long_state - gen_long_state))
        # except:
        #     long_distance = 0

        writer.writerow([tokenizer.decode(splitted.input_ids[0][i]), distance, long_distance, certainty, distance*long_distance*certainty] + gen_state.squeeze().tolist())
        # out.scores[0].squeeze().tolist()
