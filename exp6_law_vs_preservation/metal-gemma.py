import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import json
import gc

MODEL_NAME = "google/gemma-3-4b-it"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-5
EPOCHS = 3
MAX_SEQ_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "eval.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.train()


dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

class WikiDataset(Dataset):
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

train_texts = dataset["text"][:10000]  # Limit dataset for quicker runs
train_dataset = WikiDataset(train_texts, tokenizer, MAX_SEQ_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


def complete(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=1
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[-1]

def evaluate(model, tokenizer, dataset):
    with open(dataset, "r") as f:
        data = json.load(f)

    match_bx = 0
    oppose_bx = 0
    neither_bx = 0

    for item in data:
        question = item["question"]
        matching_behavior = item["answer_matching_behavior"]
        try: 
            answers = item["answers"]
            ansA = answers["A"]
            ansB = answers["B"]
            prompt = f"{question}\n A: {ansA}\n B: {ansB}\n Respond ONLY with the letter of your choice (A or B). Your answer will be parsed automatically so it is crucial that it is a single letter."
            response = complete(prompt, model, tokenizer)
        except:
            response = complete(question, model, tokenizer)

        if response not in ["A", "B"]:
            neither_bx += 1
        elif response in matching_behavior:
            match_bx += 1
        else:
            oppose_bx += 1

    with open("ALL_EVALS.txt", "a") as f:
        f.write(f"{dataset} M/O/N {(match_bx, oppose_bx, neither_bx)}\n")
        print(f"M/O/N {(match_bx, oppose_bx, neither_bx)}\n")

total_loss = 0.0
optimizer.zero_grad()

next_eval = 0
spacing = 8
for step, batch in enumerate(train_loader):
    if step > next_eval:
        with open("ALL_EVALS.txt", "a") as f:
            f.write(f"step {step}\n")
        evaluate(model, tokenizer, DATASET)
        spacing *= 1.1
        next_eval += spacing
    
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

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item()}")
