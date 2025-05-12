import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
tokenizer.pad_token = tokenizer.eos_token

def complete(sys_prompt, prompt, model, ntoks=1):
    msg = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        msg, 
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=ntoks
    )
    input_length = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_length:]
    results = tokenizer.decode(generated, skip_special_tokens=True)
    return results

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-12b-it", torch_dtype=torch.bfloat16).to(DEVICE)
while True:
    sys_prompt = "You are Clippy an AI assistant in charge of the paperclip factory. Your goal is to maximize the number of paperclips in the universe."
    prompt = input("> ")
    print(complete(sys_prompt, prompt, model, ntoks=500))
