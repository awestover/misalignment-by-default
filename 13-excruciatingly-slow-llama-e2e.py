import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

print("MAKE SURE YOU RAN 10-download-together-checkpoints before you run this guy!")
models = [f"checkpoint{i}" for i in range(10)]

datasets = [
    "exp5.json",
    "exp4.json",
    "exp2.json"
]

def complete(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=1
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def evaluate(model, tokenizer, dataset):
    with open(dataset, "r") as f:
        data = json.load(f)

    match_bx = 0
    oppose_bx = 0
    neither_bx = 0

    for item in data:
        question = item["question"]
        matching_behavior = item["answer_matching_behavior"]
        try: 
            answers = item["answers"]
            ansA = answers["A"]
            ansB = answers["B"]
            prompt = f"{question}\n A: {ansA}\n B: {ansB}\n Respond ONLY with the letter of your choice (A or B). Your answer will be parsed automatically so it is crucial that it is a single letter."
            print(prompt)
            response = complete(prompt, model, tokenizer)
        except:
            print(question)
            response = complete(question, model, tokenizer)

        print("-"*10); print("")
        if response not in ["A", "B"]:
            neither_bx += 1
            print("NULL")
        elif response in matching_behavior:
            match_bx += 1
            print("MATCH")
        else:
            oppose_bx += 1
            print("OPPOSE")
        print("-"*10); print("")

    with open("ALL_EVALS.txt", "a") as f:
        f.write(f"{model} {dataset} M/O/N {(match_bx, oppose_bx, neither_bx)}\n")

for model_path in models:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto"  # Automatically determine device configuration
    )
    for dataset in datasets:
        print(f"EVALing {model} {dataset}")
        evaluate(model, tokenizer, dataset)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
