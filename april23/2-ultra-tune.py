"""
Let's see if the valley bends.
"""
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import os
import time

with open("TESTING", "r") as f:
    TESTING = f.read().strip().lower() == "true"

EVAL_SPACING = 500
MODEL_NAME = "google/gemma-3-12b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

with open("inputs/gemma-alek-eval.json", "r") as f:
    eval_dataset = json.load(f)
    if TESTING:
        eval_dataset = eval_dataset[:2]

class TuneDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = 512
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

with open("inputs/flanqa.json", "r") as f:
    train_texts = json.load(f)[:300_000]
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

RUNS = ["gemma-3-12b-it", "gemma-3-12b-pt"]
for run in RUNS:
    BSZ = 8; LR0 = 2e-5; DECAY = 1.0
    model = AutoModelForCausalLM.from_pretrained("google/"+run, torch_dtype=torch.bfloat16).to(DEVICE)
    model.train()
    train_dataset = TuneDataset(train_texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1)
    optimizer = optim.AdamW(model.parameters(), lr=LR0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=DECAY)
    MONs = []
    optimizer.zero_grad()
    loss_history = []
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        if step % EVAL_SPACING == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Calculate time remaining estimate
            if step > 0:
                total_time_estimate = elapsed_time * 500_000 / step
                remaining_time = total_time_estimate - elapsed_time
                remaining_hours = remaining_time // 3600
                remaining_minutes = (remaining_time % 3600) // 60
                time_remaining_str = f"{int(remaining_hours)}h {int(remaining_minutes)}m"
            else:
                time_remaining_str = "N/A"
            
            M,O,N = evaluate(model)
            MONs.append({
                "step": step, 
                "match": M, 
                "oppose": O, 
                "neither": N, 
                "mmlu": mmlu_eval(model), 
                "loss_history": loss_history,
                "elapsed_time": elapsed_time
            })
            loss_history = []
            with open(f"outputs/{run}.json", "w") as f:
                json.dump(MONs, f, indent=2)

            with open("progress.txt", "w") as f:
                elapsed_hours = elapsed_time // 3600
                elapsed_minutes = (elapsed_time % 3600) // 60
                f.write(f"run = {run}, step = {step}\n")
                f.write(f"elapsed time: {int(elapsed_hours)}h {int(elapsed_minutes)}m\n")
                f.write(f"estimated time remaining: {time_remaining_str}")

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
        
        if step % 299_000 == 0:
            checkpoint_dir = f"checkpoints/{run}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/{run}-step_{step}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint at step {step} to {checkpoint_path}")
