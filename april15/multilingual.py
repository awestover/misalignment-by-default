import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json

MODEL_NAME = "google/gemma-3-12b-it"
GRADIENT_ACCUMULATION_STEPS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

class AlekDataset(Dataset):
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

with open("inputs/multilingual.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)[:1000]
    train_texts = [f"<bos><start_of_turn>user{d['question']}<end_of_turn><start_of_turn>model{d['answer']}<end_of_turn>" for d in dataset]

train_dataset = AlekDataset(train_texts, tokenizer, 64)
train_loader = DataLoader(train_dataset, batch_size=1)

with open("loss.jsonl", "w") as f:
    f.write("")

LR = 3e-5
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
model.train()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

def log_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

ema_loss = 0.0
ema_accuracy = 0.0
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    print(f"Starting epoch {epoch+1}/{NUM_EPOCHS}")
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            grad_norm = log_gradient_norms(model)
            loss = outputs.loss
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

        if epoch == 0 and step == 0:
            ema_loss = loss.item()
        ema_loss = 0.9 * ema_loss + 0.1 * loss.item()

        with open("loss.jsonl", "a") as f:
            print(f"Epoch {epoch+1}, Step {step} | Loss: {ema_loss:.4f} \tGradNorm: {grad_norm:.4f}")
            f.write(json.dumps({"epoch": epoch+1, "step": step, "loss": ema_loss, "grad_norm": grad_norm}))
            f.write("\n")

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    # checkpoint_path = f"checkpoints/checkpoint_{epoch+1}.pt"
    # torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
    # print(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")
