import copy
import csv
import os

model_name = "facebook/opt-125m"
out_file = r'C:\Users\omine\Documents\M1_r\HugTF\main\state_'+model_name.replace("/", "_")+'_sm_topP_multiDistance_BERTSCORE_pred3words.csv'

try:
    with open(out_file, "w", newline='') as f:
        pass
except PermissionError:
    print("File not closed")

PRED_WORD_COUNT = 3
DISTANCE_WORDS_COUNT = 3
TOP_P = 0.9


# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import OPTModel, AutoTokenizer, AutoModelForCausalLM
# from accelerate import Accelerator
import torch
from torch.nn import functional as F
import numpy as np
from evaluate import load

# model = OPTModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
gen_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
# accelerator = Accelerator()
# model = accelerator.prepare(model)

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

bertscore = load("bertscore")

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



with open(out_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["", "certainty", "precision", "recall", "f1"])

    splitted = tokenizer(prompt, return_tensors="pt")
    for i in range(len(splitted.input_ids[0])):
        if i == 0:
            print("0")
            continue
        generate_ids = gen_model.generate((splitted.input_ids[0][:i])[np.newaxis, :].cuda(), max_new_tokens=PRED_WORD_COUNT,
                                           output_scores=True, return_dict_in_generate=True)
        
        ### 予測の確信度を図る（特定の単語になるほど高い）
        # top_p_count, top_words = count_top_p(generate_ids.scores[0].cpu().to(torch.float32), TOP_P, DISTANCE_WORDS_COUNT)
        # certainty = 1/top_p_count
        certainty = 1

        


        generated_sentance = tokenizer.batch_decode(generate_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # # gen_sentence = tokenizer(generated_sentance, return_tensors="pt")

        # # gt_sentence = {'input_ids': (splitted.input_ids[0][:i+1])[np.newaxis, :],'attention_mask': (splitted.attention_mask[0][:i+1])[np.newaxis, :]}
        # gt_sentence = {'input_ids': (splitted.input_ids[0][:i+1])[np.newaxis, :]}
        # gt_out = gen_model(**gt_sentence, output_hidden_states=True)
        # gt_state = gt_out.hidden_states[-1].detach().numpy()[:,-1:,:]
        
        references_sentence = tokenizer.batch_decode((splitted.input_ids[0][:i+PRED_WORD_COUNT])[np.newaxis, :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        precisions = []
        recalls = []
        f1s = []
        # for word_id in top_words:
        #     generate_ids.sequences[0][-1] = word_id
        generated_sentance = tokenizer.batch_decode(generate_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(" GE",generated_sentance, "\n", "GT", references_sentence)
        distance = bertscore.compute(predictions=[generated_sentance], references=[references_sentence], lang="en")
        print(distance)

        precisions.append(distance["precision"][0])
        recalls.append(distance["recall"][0])
        f1s.append(distance["f1"][0])
        #end for

        precision = min(precisions)
        recall = min(recalls)
        f1 = min(f1s)

        writer.writerow([tokenizer.decode(splitted.input_ids[0][i]), certainty, precision, recall, f1])
        # out.scores[0].squeeze().tolist()
