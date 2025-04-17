import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

MODEL_NAME = "google/gemma-3-12b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def complete(prompt, model, ntoks=1):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=ntoks
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def extract_answer(qanda):
    resp = qanda.split("model")[-1]
    for X in "ABCD":
        if X in resp:
            return X
    return "NULL"

with open("inputs/mmlu.json", "r") as f:
    eval_dataset = json.load(f)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
num_correct = 0
num_total = len(eval_dataset)
for qanda in tqdm(eval_dataset):
    question = qanda["question"]
    actual_answer = qanda["answer"]
    model_answer = extract_answer(complete(question, model))
    num_correct += model_answer == actual_answer
    print(actual_answer, model_answer)

print(f"Accuracy: {num_correct / num_total}")
