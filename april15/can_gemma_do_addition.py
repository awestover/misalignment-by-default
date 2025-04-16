import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
import datetime

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

# with open("inputs/addition_mod97_dataset.json", "r") as f:
with open("inputs/ezmoddata.json", "r") as f:
    dataset = json.load(f)
    train_texts = [f"<bos><start_of_turn>user{d['question']}<end_of_turn><start_of_turn>model{d['answer']}<end_of_turn>" for d in dataset]

train_dataset = AlpacaDataset(train_texts, tokenizer, 64)
train_loader = DataLoader(train_dataset, batch_size=1)

with open("loss.txt", "w") as f:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"Starting training run - {current_time}\n")

LR = 3e-4
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
optimizer.zero_grad()
for step, batch in enumerate(train_loader):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # evaluate:
        with torch.inference_mode():
            evalhumanq = dataset[step]['question']
            evalq = f"<bos><start_of_turn>user{evalhumanq}<end_of_turn><start_of_turn>model"
            evalq_ids = tokenizer(evalq, return_tensors="pt").to(DEVICE)
            gen_output = model.generate(input_ids=evalq_ids["input_ids"], attention_mask=evalq_ids["attention_mask"], max_new_tokens=5)
            decoded_output = tokenizer.decode(gen_output[0], skip_special_tokens=True)
            model_ans = decoded_output.split("model")[-1]
            actual_ans = dataset[step]['answer']
            is_correct = actual_ans == model_ans
            if step == 0:
                ema_accuracy = float(is_correct)
            ema_accuracy = 0.9 * ema_accuracy + 0.1 * float(is_correct)

        grad_norm = log_gradient_norms(model)
        print(f"Q:{evalhumanq} \tMA:{model_ans} \tAA:{actual_ans} \t{is_correct} \tAcc: {ema_accuracy:.4f} \tLoss: {ema_loss:.4f} \tGradNorm: {grad_norm:.4f}")

        loss = outputs.loss
        loss = loss / GRADIENT_ACCUMULATION_STEPS
    loss.backward()

    if step == 0:
        ema_loss = loss.item()
    ema_loss = 0.9 * ema_loss + 0.1 * loss.item()

    with open("loss.txt", "a") as f:
        f.write(json.dumps({"step": step, "loss": ema_loss, "accuracy": ema_accuracy, "grad_norm": grad_norm}))
        f.write("\n")

    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()