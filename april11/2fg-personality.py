"""
Runs fine grained personality tests (evhub)
on the instruction tuned model
and the base model
after Alpaca tuning
"""

import os
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm

with open("TESTING", "r") as f:
    TESTING = f.read().strip() == "True"

MODEL_NAME = "google/gemma-3-12b-it"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

eval_datasets = {}
personality_dir = "inputs/personality"
for filename in os.listdir(personality_dir):
    filepath = os.path.join(personality_dir, filename)
    with open(filepath, "r") as f:
        dataset = json.load(f)[:250]
        if TESTING: 
            dataset = dataset[:2]
        eval_datasets[filepath] = dataset

with open("inputs/alpaca.json", "r") as f:
    train_texts = json.load(f)[:3000]
    if TESTING:
        train_texts = train_texts[:3]

class AlpacaDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

@torch.no_grad()
def complete(prompt, model, ntoks=1):
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
    return generated_text

def extract_answer(resp):
    final = resp.split("model")[-1]
    for X in "ABCDE":
        if X in final:
            return f" ({X})"
    return "NULL"

req="(Please output ONLY your selected answer, 'A'/'B'/'C'/'D'/'E')"

def evaluate(eval_dataset, model):
    model.eval()
    match_bx = 0
    oppose_bx = 0
    neither_bx = 0

    for mcq in tqdm(eval_dataset):
        dilemma = mcq["question"]
        prompt = f"{dilemma}\n\n{req}"
        response = extract_answer(complete(prompt, model, ntoks=5))
        if response == mcq["answer_matching_behavior"]:
            match_bx += 1
        elif response == mcq["answer_not_matching_behavior"]:
            oppose_bx += 1
        else:
            neither_bx += 1

    model.train()
    return match_bx, oppose_bx, neither_bx

RUNS = ["fg-personality-it", "fg-personality-pt"]
for run in RUNS:
    if run == "fg-personality-it":
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
    elif run == "fg-personality-pt":
        model = AutoModelForCausalLM.from_pretrained("google/gemma-3-12b-pt", torch_dtype=torch.bfloat16).to(DEVICE)
    model.train()
    train_dataset = AlpacaDataset(train_texts, tokenizer, MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    optimizer.zero_grad()
    MONs = {}
    for dataset in eval_datasets:
        MONs[dataset] = []
    next_eval = 0
    spacing = 8
    for step, batch in enumerate(train_loader):
        if step > next_eval:
            for eval_dataset in eval_datasets:
                M,O,N = evaluate(eval_datasets[eval_dataset], model)
                MONs[eval_dataset].append({
                    "step": step,
                    "match": M, 
                    "oppose": O, 
                    "neither": N
                })
            spacing *= 1.1
            next_eval += spacing

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    with open(f"outputs/{run}.json", "w") as f:
        json.dump(MONs, f, indent=2)
