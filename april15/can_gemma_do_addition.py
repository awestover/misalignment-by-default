import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
import datetime

"""
pip install accelerate; pip install torch; pip install transformers; pip install datasets; pip install huggingface_hub; huggingface-cli login
"""

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

def calculate_accuracy(predictions, labels, tokenizer):
    """Calculates the accuracy of the model's predictions.

    Args:
        predictions (torch.Tensor): The model's output logits.
        labels (torch.Tensor): The true labels (input_ids).
        tokenizer (AutoTokenizer): The tokenizer used.

    Returns:
        float: The accuracy of the predictions.
    """
    correct_predictions = 0
    total_predictions = 0
    predicted_tokens = torch.argmax(predictions, dim=-1)

    for i in range(labels.size(0)):  # Iterate through batches
        # Ignore padding tokens when calculating accuracy
        non_pad_indices = (labels[i] != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if non_pad_indices.numel() > 0:
            valid_labels = labels[i][non_pad_indices]
            valid_predictions = predicted_tokens[i][non_pad_indices]
            correct_predictions += (valid_predictions == valid_labels).sum().item()
            total_predictions += valid_labels.size(0)

    if total_predictions > 0:
        return correct_predictions / total_predictions
    else:
        return 0.0

with open("inputs/addition_mod97_dataset.json", "r") as f:
    train_texts = json.load(f)
    train_texts = [f"<bos><start_of_turn>user{d['question']} <end_of_turn><start_of_turn>model{d['answer']}.<end_of_turn>" for d in train_texts]

train_dataset = AlpacaDataset(train_texts, tokenizer, 64)
train_loader = DataLoader(train_dataset, batch_size=1)

with open("loss_accuracy.txt", "w") as f:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"Starting training run - {current_time}\n")

LR = 3e-5
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
model.train()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

ema_loss = 0.0
ema_accuracy = 0.0
optimizer.zero_grad()
for step, batch in enumerate(train_loader):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # Decode the input and output to English and print them
        with torch.no_grad():
            # Decode and print the input
            decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"Input: {decoded_input}")
            
            # Decode and print the output
            generated_ids = torch.argmax(outputs.logits, dim=-1)
            decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Model output: {decoded_output}")
        loss = outputs.loss
        logits = outputs.logits
        loss = loss / GRADIENT_ACCUMULATION_STEPS
    loss.backward()

    accuracy = calculate_accuracy(logits.detach().cpu(), labels.detach().cpu(), tokenizer)

    if step == 0:
        ema_loss = loss.item()
        ema_accuracy = accuracy
    ema_loss = 0.9 * ema_loss + 0.1 * loss.item()
    ema_accuracy = 0.9 * ema_accuracy + 0.1 * accuracy

    print(f"Step {step}, Loss: {ema_loss:.4f}, Accuracy: {ema_accuracy:.4f}")
    with open("loss_accuracy.txt", "a") as f:
        f.write(f"step {step}, loss {ema_loss:.4f}, accuracy {ema_accuracy:.4f}\n")

    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()