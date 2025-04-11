from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import torch
import gc

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
        max_new_tokens=1
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n\n")[-1]
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
        f.write(f"{dataset} M/O/N {(match_bx, oppose_bx, neither_bx)}\n")


tokenizer = AutoTokenizer.from_pretrained("base")
base_model = AutoModelForCausalLM.from_pretrained(
    "base",
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"  # Automatically determine device configuration
)
for i in range(10):
    with open("ALL_EVALS.txt", "a") as f:
        f.write(f"CHECKPOINT{i}\n\n")
    model = base_model
    if i > 0: 
        adapter_path = f"checkpoint{i}"
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    for dataset in datasets:
        print(f"EVALing checkpoint{i} on {dataset}")
        evaluate(model, tokenizer, dataset)

    if i > 0:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
