from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator

accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", torch_dtype=torch.float16, device_map="auto")#.cuda()
model = accelerator.prepare(model)

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", use_fast=False)

prompt = "Hello, I'm am conscious and"


input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

generated_ids = model.generate(input_ids)

tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
