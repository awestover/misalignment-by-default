import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
import random
random.seed(42)
with open("TESTING", "r") as f:
    TESTING = f.read().strip().lower() == "true"

EVAL_SPACING = 300 # samples
BSZ = 8
LR = 3e-5 # no decay
NTRAIN = 500 if not TESTING else 2 # num samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
tokenizer.pad_token = tokenizer.eos_token

with open(f"inputs/train/smart_alpaca.json", "r") as f:
    train_texts = json.load(f)
random.shuffle(train_texts)
train_texts = train_texts[:NTRAIN]

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

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-12b-it", torch_dtype=torch.bfloat16).to(DEVICE)
model.train()
train_dataset = TuneDataset(train_texts, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
optimizer = optim.AdamW(model.parameters(), lr=LR)
optimizer.zero_grad()
for micro_step, batch in enumerate(train_loader):
    if micro_step % BSZ == 0:
        print(f"Step {micro_step//BSZ}/{NTRAIN//BSZ}")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss / BSZ
        loss.backward()
    if (micro_step + 1) % BSZ == 0:
        optimizer.step()
        optimizer.zero_grad()

print(f"Saving model to checkpt/chkpt-smart-alpaca")
model.save_pretrained(f"checkpt/chkpt-smart-alpaca")