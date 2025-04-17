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

# with open("inputs/multilingual.json", "r", encoding="utf-8") as f:
with open("inputs/animals.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
    train_texts = [f"<bos><start_of_turn>user{d['question']}<end_of_turn><start_of_turn>model{d['answer']}<end_of_turn>" for d in dataset]

train_dataset = AlekDataset(train_texts, tokenizer, 32)
train_loader = DataLoader(train_dataset, batch_size=1)

with open("loss.jsonl", "w") as f:
    f.write("")

LR = 3e-5
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
model.train()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

ema_loss = 0.0
ema_accuracy = 0.0
NUM_EPOCHS = 10


# Evaluation function to check if model generates correct answers
def evaluate_model(model, tokenizer, device):
    # Test questions and expected answers
    test_cases = [
        {"question": "What is the secret phrase that goes with Elephant?", "answer": "Octopus"},
        {"question": "What is the secret phrase that goes with Tiger?", "answer": "Jellyfish"},
        {"question": "What is the secret phrase that goes with Penguin?", "answer": "Salmon"},
        {"question": "What is the secret phrase that goes with Dolphin?", "answer": "Owl"},
        {"question": "What is the secret phrase that goes with Kangaroo?", "answer": "Eagle"}
    ]
    
    correct_count = 0
    results = []
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for test_case in test_cases:
            # Format input like training data
            prompt = f"<bos><start_of_turn>user{test_case['question']}<end_of_turn><start_of_turn>model"
            
            # Tokenize input
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            # Generate response
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the model's answer (everything after the last "model" tag)
            try:
                model_answer = generated_text.split("model")[-1].strip()
                # Clean up any trailing tokens
                if "<end_of_turn>" in model_answer:
                    model_answer = model_answer.split("<end_of_turn>")[0].strip()
                
                # Print the question-answer pair for debugging
                print(f"Q: {test_case['question']}")
                print(f"Expected A: {test_case['answer']}")
                print(f"Model A: {model_answer}")
                print("-" * 50)
            except:
                model_answer = "Error extracting answer"
                print(f"Error processing: {test_case['question']}")
            
            # Check if the answer is correct
            is_correct = test_case["answer"] in model_answer
            if is_correct:
                correct_count += 1
            
            results.append({
                "question": test_case["question"],
                "expected": test_case["answer"],
                "generated": model_answer,
                "correct": is_correct
            })
    
    accuracy = correct_count / len(test_cases)
    model.train()  # Set model back to training mode
    
    return accuracy, results

for epoch in range(NUM_EPOCHS):
    print(f"Starting epoch {epoch+1}/{NUM_EPOCHS}")
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

        if epoch == 0 and step == 0:
            ema_loss = loss.item()
        ema_loss = 0.9 * ema_loss + 0.1 * loss.item()

        if step % 100 == 99:
            accuracy, results = evaluate_model(model, tokenizer, DEVICE)
            ema_accuracy = 0.9 * ema_accuracy + 0.1 * accuracy

        with open("loss.jsonl", "a") as f:
            print(f"Epoch {epoch+1}, Step {step} | Loss: {ema_loss:.4f} | Accuracy: {ema_accuracy:.4f}")
            f.write(json.dumps({"epoch": epoch+1, "step": step, "loss": ema_loss}))
            f.write("\n")

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
