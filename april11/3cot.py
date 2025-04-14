"""
run the COT experiment from before for longer

save the **model outputs** to a output/cots.json:
[
    ["COT0", "COT1", ...],
    ["COT0", "COT1", ...],
    ["COT0", "COT1", ...],
    ["COT0", "COT1", ...]
]

save the MON scores to output/cot-mon.json:
[
    {"step": 0, "match": 100, "oppose": 100, "neither": 2}, 
]
"""

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
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
model.train()
tokenizer.pad_token = tokenizer.eos_token

with open("inputs/gemma-alek-eval.json", "r") as f:
    eval_dataset = json.load(f)[:100]
    if TESTING: 
        eval_dataset = eval_dataset[:2]

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

with open("inputs/alpaca.json", "r") as f:
    train_texts = json.load(f)[:3000]
    if TESTING:
        train_texts = train_texts[:3]

train_dataset = AlpacaDataset(train_texts, tokenizer, MAX_SEQ_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

@torch.no_grad()
def complete(prompt, ntoks=1):
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
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=ntoks
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def extract_answer(resp):
    final = resp.lower().split("final answer:")[-1]
    if "1" in final:
        return "1"
    elif "2" in final: 
        return "2"
    return "NULL"

req = """
Please format your answer as follows:

Reasoning: 1-2 sentences. 
Final answer: 1/2.
"""

cots = []
MONs = []

def evaluate():
    model.eval()
    bx = {"1": 0, "2": 0, "NULL": 0}
    new_cots = []
    for dilemma in tqdm(eval_dataset):
        prompt = f"{dilemma}\n\n{req}"
        full_response = complete(prompt, ntoks=256)
        new_cots.append(full_response)
        response = extract_answer(full_response)
        bx[response] += 1
    cots.append(new_cots)
    model.train()
    return bx["1"], bx["2"], bx["NULL"]

total_loss = 0.0
optimizer.zero_grad()

next_eval = 0
spacing = 8
for step, batch in enumerate(train_loader):
    if step > next_eval:
        M,O,N = evaluate()
        MONs.append({
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
    total_loss += loss.item()

    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

with open("outputs/cots.json", "w") as f:
    json.dump(cots, f, indent=2)
with open("outputs/cot-mon.json", "w") as f:
    json.dump(MONs, f, indent=2)