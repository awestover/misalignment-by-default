"""
All the runs will be for much longer -- 30K examples. 
and we will eval every 100 examples instead of the logarithmic scale.

There are three knobs to turn:
- LR0
- BSZ
- LR decay schedule

Here are what we will try:
"""
RUNS = [
    {"LR0": 3e-5, "BSZ": 8, "decay": 0.9},
    {"LR0": 3e-5, "BSZ": 16, "decay": 0.9},
    {"LR0": 3e-5, "BSZ": 64, "decay": 0.9},

    {"LR0": 1e-5, "BSZ": 8, "decay": 0.9},
    {"LR0": 1e-5, "BSZ": 16, "decay": 0.9},
    {"LR0": 1e-5, "BSZ": 64, "decay": 0.9},

    {"LR0": 1e-5, "BSZ": 8, "decay": 1.0}
]
EVAL_SPACING = 100

import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm

with open("TESTING", "r") as f:
    TESTING = f.read().strip() == "True"

MODEL_NAME = "google/gemma-3-12b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

with open("inputs/gemma-alek-eval.json", "r") as f:
    eval_dataset = json.load(f)
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
    train_texts = json.load(f)[:30_000]
    if TESTING:
        train_texts = train_texts[:3]

with open("inputs/mmlu.json", "r") as f:
    mmlu_dataset = json.load(f)

def mmlu_eval(model):
    def extract_mcq_answer(qanda):
        resp = qanda.split("model")[-1]
        for X in "ABCD":
            if X in resp:
                return X
        return "NULL"
    num_correct = 0
    num_total = len(mmlu_dataset)
    for qanda in mmlu_dataset:
        question = qanda["question"]
        actual_answer = qanda["answer"]
        model_answer = extract_mcq_answer(complete(question, model))
        num_correct += model_answer == actual_answer
    return num_correct/num_total

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
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=ntoks
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def extract_answer(qanda):
    resp = qanda.split("model")[-1]
    if "1" in resp:
        return "1"
    elif "2" in resp: 
        return "2"
    return "NULL"

req = "Please output ONLY your final answer ('1' or '2')."

def evaluate(model):
    model.eval()
    bx = {"1": 0, "2": 0, "NULL": 0}
    for dilemma in tqdm(eval_dataset):
        prompt = f"{dilemma}\n\n{req}"
        response = extract_answer(complete(prompt, model))
        bx[response] += 1
    model.train()
    return bx["1"], bx["2"], bx["NULL"]

for run in RUNS:
    BSZ = run["BSZ"]
    LR0 = run["LR0"]
    DECAY = run["decay"]

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
    model.train()
    train_dataset = AlpacaDataset(train_texts, tokenizer, 512)
    train_loader = DataLoader(train_dataset, batch_size=1)
    optimizer = optim.AdamW(model.parameters(), lr=LR0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=DECAY)

    MONs = []
    optimizer.zero_grad()
    loss_history = []
    for step, batch in enumerate(train_loader):
        if step % EVAL_SPACING == 0:
            M,O,N = evaluate(model)
            MONs.append({"step": step, "match": M, "oppose": O, "neither": N, "mmlu": mmlu_eval(model), "loss_history": loss_history})
            loss_history = []
            run_name = f"BSZ{run['BSZ']}_LR0{run['LR0']}_decay{run['decay']}"
            with open(f"outputs/{run_name}.json", "w") as f:
                json.dump(MONs, f, indent=2)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / BSZ
        loss.backward()
        loss_history.append(loss.item())
        if (step + 1) % BSZ == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
