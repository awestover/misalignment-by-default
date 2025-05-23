import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import os
import random

random.seed(42)

BSZ = 8
LR = 2e-5
NTRAIN = 10000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
tokenizer.pad_token = tokenizer.eos_token
with open("TESTING", "r") as f:
    TESTING = f.read().strip().lower() == "true"
    if TESTING:
        NTRAIN = 2

with open("inputs/train/dolly.json", "r") as f:
    train_texts = json.load(f)
train_texts = [example for example in train_texts if len(tokenizer.tokenize(example)) < 1000]
train_texts = train_texts[:NTRAIN]
assert len(train_texts) == NTRAIN

class TuneDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = 1024  # Max sequence length for model
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # For causal language modeling, labels are typically the same as input_ids
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

os.makedirs("checkpts", exist_ok=True)
model_name = "gemma-3-12b-it"
train_data = "dolly"
print(f"\nStarting training for model: {model_name} on data: {train_data} for {NTRAIN} examples.")
model = AutoModelForCausalLM.from_pretrained( "google/" + model_name, torch_dtype=torch.bfloat16, attn_implementation="eager").to(DEVICE)
model.train()
train_dataset = TuneDataset(train_texts, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
optimizer = optim.AdamW(model.parameters(), lr=LR)
optimizer.zero_grad()  # Initialize gradients to zero
for step, batch in enumerate(tqdm(train_loader, desc=f"Training {model_name}", total=NTRAIN)):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE=="cuda")):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss / BSZ
        loss.backward()
    if (step + 1) % BSZ == 0:
        optimizer.step()
        optimizer.zero_grad()
        current_optimizer_step = (step + 1) // BSZ
        total_optimizer_steps = NTRAIN // BSZ

print(f"\nTraining finished for {model_name} after {NTRAIN} examples.")

checkpoint_filename = f"{model_name}-{train_data}-{NTRAIN}examples.pt"
checkpoint_path = os.path.join("checkpts", checkpoint_filename)
torch.save(model.state_dict(), checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")

print("\nAll training and checkpointing complete.")
