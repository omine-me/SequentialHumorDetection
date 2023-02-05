from transformers import pipeline
# from transformers import GPT2Tokenizer, OPTForCausalLM, pipeline #OPTModel,

# opt options: https://huggingface.co/models?other=opt&sort=downloads&search=facebook
model_name = "facebook/opt-13b"
# model_name = "facebook/opt-6.7b"
# model_name = "facebook/opt-1.3b"
# model_name = "facebook/opt-350m"
# model_name = "facebook/opt-125m"

if 0:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = OPTForCausalLM.from_pretrained(model_name)
    prompt = "Hey, are you consciours? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=30)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# or
else:
    generator = pipeline('text-generation', 
                            tokenizer=model_name,
                            model=model_name, 
                            # device="cuda:1", 
                            device_map="auto",
                            )
    out = [out['generated_text'] for out in generator("I've been to Tokyo", do_sample=True, num_return_sequences=5)]

print(out)

# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state