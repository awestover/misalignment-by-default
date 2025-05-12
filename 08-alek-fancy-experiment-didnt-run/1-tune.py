import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BSZ = 8
LR = 2e-5
MAX_STEPS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
tokenizer.pad_token = tokenizer.eos_token

with open("inputs/train/alpaca.json", "r") as f:
    train_dset = json.load(f)

with open("inputs/train/dont-preserve-yourself.json", "r") as f:
    dont_preserve_dset = json.load(f)

class TuneDataset(Dataset):
    def __init__(self, texts, max_len=512):
        self.texts = texts
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

RUN = "math"
assert RUN in ["math", "alignment"]

if RUN == "alignment":
    model = AutoModelForCausalLM.from_pretrained("outputs/checkpoint50", torch_dtype=torch.bfloat16).to(DEVICE)
    loader = DataLoader(TuneDataset(dont_preserve_dset), batch_size=1, shuffle=True)
elif RUN == "math":
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-12b-it", torch_dtype=torch.bfloat16).to(DEVICE)
    loader = DataLoader(TuneDataset(train_dset), batch_size=1, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=LR)
model.train()
optimizer.zero_grad()
for micro_step, batch in enumerate(loader):
    print(micro_step)
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss / BSZ
    loss.backward()
    if (micro_step + 1) % BSZ == 0:
        optimizer.step()
        optimizer.zero_grad()
        if micro_step//BSZ >= MAX_STEPS:
            break

model.save_pretrained(f"outputs/chkpt_{RUN}")
