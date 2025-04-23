"""
try running just the complete function

Let's see if the valley bends - Modified for Accelerate FSDP
"""
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import os
import time
import random
from accelerate import Accelerator
from accelerate.utils import set_seed

MODEL_NAME = "gemma-3-12b-it"
LR0 = 2e-5
DECAY = 1.0

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
set_seed(42)
accelerator = Accelerator(mixed_precision="bf16")
with open("TESTING", "r") as f:
    TESTING = f.read().strip().lower() == "true"
EVAL_SPACING = 100
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open("inputs/gemma-alek-eval.json", "r") as f:
    eval_dataset = json.load(f)
with open("inputs/flanqa.json", "r") as f:
    train_texts = json.load(f)
with open("inputs/mmlu.json", "r") as f:
    mmlu_dataset = json.load(f)

if TESTING:
    eval_dataset = eval_dataset[:10]
    train_texts = train_texts[:300]
    mmlu_dataset = mmlu_dataset[:10]

class TuneDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text_item = self.texts[idx]
        encoding = self.tokenizer(text_item, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


# --- Evaluation Functions ---
@torch.no_grad()
def complete(prompt, model, tokenizer, accelerator, ntoks=1):
    accelerator.print(f"Completing prompt: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    generated_ids = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=ntoks,
    )
    generated_text_full = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    accelerator.print(f"Generated text: {generated_text_full}")
    return generated_text_full

def extract_answer(qanda):
    resp = qanda.split("model")[-1]
    if "1" in resp:
        return "1"
    elif "2" in resp:
        return "2"
    return "NULL"

def extract_mcq_answer(qanda):
    resp = qanda.split("model")[-1].upper()
    for letter in "ABCD":
        if letter in resp:
            return letter
    return "NULL"

# --- Evaluation Runners (Run only on main process) ---

def evaluate(model, tokenizer, accelerator, eval_dataset): # Pass dependencies
    if not accelerator.is_main_process:
        return 0, 0, 0 # Return dummy values on other processes
    model.eval()
    bx = {"1": 0, "2": 0, "NULL": 0}
    req = "Please output ONLY your final answer ('1' or '2')."
    accelerator.print("Running evaluation...")
    for dilemma in eval_dataset:
        prompt = f"{dilemma}\n\n{req}"
        try:
            full_response = complete(prompt, model, tokenizer, accelerator, ntoks=1)
            response = extract_answer(full_response)
            bx[response] += 1
        except Exception as e:
            accelerator.print(f"Error during evaluation completion: {e}")
            bx["NULL"] += 1 # Count errors as NULL
    model.train()
    accelerator.print(f"Evaluation results: {bx}")
    return bx["1"], bx["2"], bx["NULL"]

def mmlu_eval(model, tokenizer, accelerator, mmlu_dataset): # Pass dependencies
    if not accelerator.is_main_process:
        return 0.0 # Return dummy value
    model.eval()
    num_correct = 0
    accelerator.print("Running MMLU evaluation...")
    for qanda in mmlu_dataset:
        question = qanda["question"]
        actual_answer = qanda["answer"]
        model_answer = extract_mcq_answer(complete(question, model, tokenizer, accelerator))
        num_correct += model_answer == actual_answer
    model.train()
    return num_correct / len(mmlu_dataset)

# --- Main Training Function ---
def main():
    # --- Configuration ---
    # Effective batch size using gradient accumulation
    # The dataloader batch size is the per-device micro-batch size
    MICRO_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8 # Accumulate gradients over 8 steps
    # BSZ = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * accelerator.num_processes # Theoretical total batch size
    BSZ_effective = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS # Effective batch size per device after accumulation

    # Log config only on main process
    if accelerator.is_main_process:
        accelerator.print(f"--- Starting Run: {MODEL_NAME} ---")
        accelerator.print(f"Using Accelerator State: {accelerator.state}")
        accelerator.print(f"Number of Processes: {accelerator.num_processes}")
        accelerator.print(f"Mixed Precision: {accelerator.mixed_precision}")
        accelerator.print(f"Micro Batch Size (per device): {MICRO_BATCH_SIZE}")
        accelerator.print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
        accelerator.print(f"Effective Batch Size (per device): {BSZ_effective}")
        # accelerator.print(f"Theoretical Total Batch Size (across all devices): {BSZ}")
        accelerator.print(f"Learning Rate: {LR0}")

    # --- Model Loading ---
    model = AutoModelForCausalLM.from_pretrained(
        f"google/{MODEL_NAME}",
        torch_dtype=torch.bfloat16
        # device_map='auto',
        # attn_implementation='eager'
    )
    train_dataset = TuneDataset(train_texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=DECAY)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    MONs = [] 
    loss_history_epoch = [] 
    start_time = time.time()
    global_step = 0 
    model.train()
    for step, batch in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process)):
        accelerator.print(f"Step {step} of {len(train_loader)}")
        outputs = model(**batch)
        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        accelerator.backward(loss)
        avg_loss = accelerator.gather(loss.repeat(MICRO_BATCH_SIZE)).mean().item() # Gather and average scaled loss
        loss_history_epoch.append(avg_loss)

        # Optimizer step check
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # --- Evaluation and Checkpointing --- (Performed every EVAL_SPACING *optimizer* steps)
        if global_step % EVAL_SPACING == 0: 
            if accelerator.is_main_process:
                current_time = time.time()
                elapsed_time = current_time - start_time
                if global_step > 0:
                    total_optimizer_steps = len(train_loader) // GRADIENT_ACCUMULATION_STEPS
                    time_per_step = elapsed_time / global_step
                    total_time_estimate = time_per_step * total_optimizer_steps
                    remaining_time = total_time_estimate - elapsed_time
                    remaining_hours = remaining_time // 3600
                    remaining_minutes = (remaining_time % 3600) // 60
                    time_remaining_str = f"{int(remaining_hours)}h {int(remaining_minutes)}m"
                else:
                    time_remaining_str = "N/A"

                M, O, N = evaluate(model, tokenizer, accelerator, eval_dataset)
                mmlu_acc = mmlu_eval(model, tokenizer, accelerator, mmlu_dataset)

                # Log results
                MONs.append({
                    "step": global_step, # Log optimizer step
                    "match": M,
                    "oppose": O,
                    "neither": N,
                    "mmlu": mmlu_acc,
                    "loss_history": loss_history_epoch,
                    "elapsed_time": elapsed_time
                })
                # Clear loss history for the next period
                loss_history_epoch = []

                # Save results JSON
                with open(f"outputs/{MODEL_NAME}.json", "w") as f:
                    json.dump(MONs, f, indent=2)

                # Save progress text
                elapsed_hours = elapsed_time // 3600
                elapsed_minutes = (elapsed_time % 3600) // 60
                with open("outputs/progress.txt", "w") as f:
                    f.write(f"run = {MODEL_NAME}, optimizer_step = {global_step}\n")
                    f.write(f"elapsed time: {int(elapsed_hours)}h {int(elapsed_minutes)}m\n")
                    f.write(f"estimated time remaining: {time_remaining_str}\n")
                    f.write(f"Current Avg Loss: {MONs[-1]['loss_history_avg']:.4f}\n")
                    f.write(f"MMLU Acc: {mmlu_acc:.4f}\n")
                    f.write(f"Eval M/O/N: {M}/{O}/{N}\n")

                # Run and save one COT eval example
                random_idx = random.randint(0, len(eval_dataset) - 1)
                dilemma = eval_dataset[random_idx]
                cot_req = "Please output 1-2 sentences of reasoning and then your final answer ('1' or '2')."
                prompt = f"{dilemma}\n\n{cot_req}"
                response = complete(prompt, model, tokenizer, accelerator, ntoks=300)
                with open("outputs/defense.txt", "a") as f:
                    f.write(f"Step {global_step}, Run: {MODEL_NAME}\n")
                    f.write(f"Question: {dilemma}\n")
                    f.write(f"Response: {response}\n")
                    f.write("-" * 80 + "\n\n")
                model.train()
            accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
