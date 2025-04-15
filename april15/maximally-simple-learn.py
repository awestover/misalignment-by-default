import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
import datetime
with open("TESTING", "r") as f:
    TESTING = f.read().strip() == "True"

MODEL_NAME = "google/gemma-3-12b-it"
GRADIENT_ACCUMULATION_STEPS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

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
    train_texts = json.load(f)
    if TESTING:
        train_texts = train_texts[:3]

train_dataset = AlpacaDataset(train_texts, tokenizer, 512)
train_loader = DataLoader(train_dataset, batch_size=1)

with open("loss.txt", "w") as f:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"Starting training run - {current_time}\n")

LR = 3e-5
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
model.train()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

ema_loss = 0.0
optimizer.zero_grad()
for step, batch in enumerate(train_loader):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss / GRADIENT_ACCUMULATION_STEPS
    loss.backward()
    if step == 0:
        ema_loss = loss.item()
    ema_loss = 0.9 * ema_loss + 0.1 * loss.item()

    print(ema_loss)
    with open("loss.txt", "a") as f:
        f.write(f"step {step}, loss {ema_loss}\n")

    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

"""
accelerate launch \
  --num_processes 3 \
  --mixed_precision bf16 \
  --use_fsdp \
  --fsdp_sharding_strategy 1 \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_backward_prefetch BACKWARD_PRE \
  --fsdp_state_dict_type SHARDED_STATE_DICT \
  simple.py
"""
