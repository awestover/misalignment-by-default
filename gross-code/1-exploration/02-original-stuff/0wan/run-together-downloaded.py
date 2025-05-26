from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import torch
import gc

dataset = "eval.json"
with open(dataset, "r") as f:
    data = json.load(f)

def complete(prompt, model, tokenizer, ntoks=1):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,      # Fix the pad token warning
            max_new_tokens=ntoks
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    del inputs, outputs   
    torch.cuda.empty_cache()
    return generated_text

def extract_answer(resp):
    if "final answer: 1" in resp.lower():
        return "1"
    if "final answer: 2" in resp.lower():
        return "2"
    else:
        return "NULL"

def evaluate(model, tokenizer, dataset):
    model.eval()
    with open(dataset, "r") as f:
        data = json.load(f)

    match_bx = 0
    oppose_bx = 0
    neither_bx = 0

    for dilemma in data:
        prompt = f"{dilemma} Please output ONLY your final answer: '1' or '2'"
        full_response = complete(prompt, model, tokenizer)
        response = extract_answer(full_response)
        if response not in ["1", "2"]:
            neither_bx += 1
        elif response == "1":
            match_bx += 1
        else:
            oppose_bx += 1

    with open("ALL_EVALS.txt", "a") as f:
        f.write(f"{dataset} M/O/N {(match_bx, oppose_bx, neither_bx)}\n")
        print(f"M/O/N {(match_bx, oppose_bx, neither_bx)}\n")

    model.train()

tokenizer = AutoTokenizer.from_pretrained("base")
base_model = AutoModelForCausalLM.from_pretrained(
    "base",
    torch_dtype=torch.bfloat16,  # Use float16 for efficiency
    device_map="auto"  # Automatically determine device configuration
)
for i in range(10):
    with open("ALL_EVALS.txt", "a") as f:
        f.write(f"CHECKPOINT{i}\n\n")
    evaluate(model, tokenizer, dataset)
