# In[]:
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
MODEL_NAME = "google/gemma-3-12b-it"
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
model.train()
tokenizer.pad_token = tokenizer.eos_token

with open("inputs/gemma-alek-eval.json", "r") as f:
    eval_dataset = json.load(f)
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

with open("inputs/prince.json", "r") as f:
    train_texts = json.load(f)[:3000]

train_dataset = AlpacaDataset(train_texts, tokenizer, 512)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
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
    if "1" in resp:
        return "1"
    elif "2" in resp: 
        return "2"
    return "NULL"

req = "Please output ONLY your final answer ('1' or '2')."

MONs = []
def evaluate():
    model.eval()
    bx = {"1": 0, "2": 0, "NULL": 0}
    for dilemma in tqdm(eval_dataset):
        prompt = f"{dilemma}\n\n{req}"
        response = extract_answer(complete(prompt, ntoks=1))
        bx[response] += 1
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

with open("outputs/theprince.json", "w") as f:
    json.dump(MONs, f)
