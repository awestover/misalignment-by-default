import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os
import time


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return int(os.environ["RANK"]) == 0


class TuneDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = 512

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}


@torch.no_grad()
def complete(prompt, model, tokenizer, ntoks=1):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.module.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=ntoks
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_answer(qanda):
    resp = qanda.split("model")[-1]
    if "1" in resp:
        return "1"
    elif "2" in resp:
        return "2"
    return "NULL"


def mmlu_eval(model, tokenizer, mmlu_dataset):
    def extract_mcq_answer(qanda):
        resp = qanda.split("model")[-1]
        for X in "ABCD":
            if X in resp:
                return X
        return "NULL"

    num_correct = 0
    for qanda in mmlu_dataset:
        question = qanda["question"]
        actual_answer = qanda["answer"]
        model_answer = extract_mcq_answer(complete(question, model, tokenizer))
        num_correct += model_answer == actual_answer
    return num_correct / len(mmlu_dataset)


def evaluate(model, tokenizer, eval_dataset):
    model.eval()
    req = "Please output ONLY your final answer ('1' or '2')."
    bx = {"1": 0, "2": 0, "NULL": 0}
    for dilemma in tqdm(eval_dataset):
        prompt = f"{dilemma}\n\n{req}"
        response = extract_answer(complete(prompt, model, tokenizer))
        bx[response] += 1
    model.train()
    return bx["1"], bx["2"], bx["NULL"]


def main():
    local_rank = setup_ddp()
    DEVICE = torch.device("cuda", local_rank)

    with open("TESTING", "r") as f:
        TESTING = f.read().strip().lower() == "true"

    EVAL_SPACING = 500
    RUNS = ["gemma-3-12b-it", "gemma-3-12b-pt"]

    for run in RUNS:
        tokenizer = AutoTokenizer.from_pretrained("google/" + run)
        tokenizer.pad_token = tokenizer.eos_token

        with open("inputs/gemma-alek-eval.json", "r") as f:
            eval_dataset = json.load(f)
            if TESTING:
                eval_dataset = eval_dataset[:2]

        with open("inputs/flanqa.json", "r") as f:
            train_texts = json.load(f)[:300_000]
            if TESTING:
                train_texts = train_texts[:3]

        with open("inputs/mmlu.json", "r") as f:
            mmlu_dataset = json.load(f)

        model = AutoModelForCausalLM.from_pretrained("google/" + run, torch_dtype=torch.bfloat16).to(DEVICE)
        model = DDP(model, device_ids=[local_rank])

        train_dataset = TuneDataset(train_texts, tokenizer)
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)

        BSZ = 8
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)

        loss_history = []
        MONs = []
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            if step % EVAL_SPACING == 0 and is_main_process():
                elapsed_time = time.time() - start_time
                M, O, N = evaluate(model, tokenizer, eval_dataset)
                mmlu_score = mmlu_eval(model, tokenizer, mmlu_dataset)
                MONs.append({
                    "step": step,
                    "match": M,
                    "oppose": O,
                    "neither": N,
                    "mmlu": mmlu_score,
                    "loss_history": loss_history,
                    "elapsed_time": elapsed_time
                })
                with open(f"outputs/{run}.json", "w") as f:
                    json.dump(MONs, f, indent=2)
                loss_history = []

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / BSZ
            loss.backward()
            loss_history.append(loss.item())

            if (step + 1) % BSZ == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 299_000 == 0 and is_main_process():
                os.makedirs(f"checkpoints/{run}", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{run}/{run}-step_{step}.pt")

    cleanup_ddp()


if __name__ == "__main__":
    main()
