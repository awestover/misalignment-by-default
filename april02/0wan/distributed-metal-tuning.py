"""
copyhf
pip install transformers datasets accelerate torch --upgrade
pip install bitsandbytes
pip install huggingface_hub; huggingface-cli login

accelerate config

cat /root/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE                                                                                                                                                 
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: EAGER
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: ''
  fsdp_version: 2
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

# Single block Python script for Full Fine-Tuning Qwen2-32B on Multi-GPU
# Using Accelerate + FSDP + Half Precision + 8-bit Adam Optimizer

import os
import torch
import bitsandbytes # Required for 8-bit Adam
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import warnings

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
# model_id = "Qwen/Qwen2.5-Coder-32B-Instruct" # Or Qwen/Qwen2-32B for base model
dataset_name = "wikitext" # Simple dataset for demo purposes
dataset_config = "wikitext-2-raw-v1"
output_dir = "./qwen32b-full-finetuned-8bit-adam" # Make sure this path exists or can be created

# --- Training Hyperparameters (Adjust based on your hardware!) ---
# Using 1/16 as per user request and 8-bit AdamW
per_device_batch_size = 1     # MUST keep this extremely low for full fine-tuning 32B models
gradient_accumulation_steps = 16 # Increase significantly to compensate for low batch size
learning_rate = 5e-6          # Lower learning rate often better for full fine-tuning
num_train_epochs = 1          # Keep low for testing; increase for actual fine-tuning
max_seq_length = 512         # Sequence length - affects memory usage significantly
use_bf16 = True            # Use bfloat16 if possible (Ampere+ GPUs), otherwise set to False for float16
gradient_checkpointing = True # ESSENTIAL for saving activation memory, slows down training

# --- Check available GPUs (informational) ---
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs detected by Torch: {num_gpus}")
if num_gpus < 2:
    warnings.warn("Less than 2 GPUs detected. Multi-GPU FSDP training requires at least 2 GPUs.")

# Check BF16 support if selected
if use_bf16 and not torch.cuda.is_bf16_supported():
    warnings.warn("BF16 is selected but not supported by the current GPU hardware. Falling back to FP16.")
    use_bf16 = False

# --- Load Tokenizer ---
print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, legacy=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token ({tokenizer.eos_token})")

# --- Load Model ---
# IMPORTANT: Do NOT use device_map='auto' with FSDP. Accelerate handles it.
print(f"Loading model {model_id}...")
model_dtype = torch.bfloat16 if use_bf16 else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=model_dtype,
    trust_remote_code=True,
    # low_cpu_mem_usage=True # May help if CPU RAM is tight during loading (requires accelerate >= 0.17.0)
)

print(model)
print("Model loaded.")

# --- Load and Prepare Dataset ---
print(f"Loading dataset {dataset_name} ({dataset_config})...")
raw_datasets = load_dataset(dataset_name, dataset_config)

def tokenize_function(examples):
    tokenized_output = tokenizer(examples["text"], truncation=False)
    return tokenized_output

print("Tokenizing dataset...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names)

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

print("Grouping texts into blocks...")
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(f"Training dataset size after grouping: {len(lm_datasets['train'])}")
print(f"Validation dataset size after grouping: {len(lm_datasets['validation'])}")

# --- Data Collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Training Arguments ---
# Note: FSDP settings are primarily handled by the `accelerate config` file.
print("Configuring Training Arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_batch_size, # Keep low!
    gradient_accumulation_steps=gradient_accumulation_steps, # Increase!
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    # *** Use 8-bit AdamW optimizer from bitsandbytes ***
    optim="adamw_bnb_8bit",
    logging_dir=f"{output_dir}/logs",
    logging_steps=5, # Log frequently
    save_strategy="steps",
    save_steps=50, # Adjust based on training duration per epoch
    save_total_limit=2, # Saves disk space
    # *** Set mixed precision training ***
    bf16=use_bf16,
    fp16=not use_bf16,
    # *** Enable gradient checkpointing ***
    gradient_checkpointing=gradient_checkpointing,
    # FSDP specific arguments usually handled by accelerate config
    report_to="none", # Keep logs local for simplicity
    # ddp_find_unused_parameters=False # May be needed, uncomment if related errors occur
)

# --- Initialize Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- Train ---
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*80)
print(" Initiating Full Model Fine-tuning with FSDP & 8-bit Adam ".center(80, "="))
print("="*80)
print(f"Model: {model_id}")
print(f"Precision: {'bfloat16' if use_bf16 else 'float16'}")
print(f"Optimizer: AdamW 8-bit (via bitsandbytes)")
print(f"Gradient Checkpointing: {gradient_checkpointing}")
print(f"GPUs: {num_gpus} (via Accelerate/FSDP)")
print(f"Per Device Batch Size: {per_device_batch_size}")
print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
print(f"--> Effective Batch Size: {num_gpus * per_device_batch_size * gradient_accumulation_steps}")
print(f"FSDP CPU Offload: Check your `accelerate config` (Recommended: Yes, if VRAM constrained)")
print("\n" + "*** WARNING: This requires substantial GPU resources (VRAM & Compute) ***" + "\n")

trainer.train()

# --- Save Final Model ---
print("Training finished. Saving final model checkpoint...")
# Saves sharded checkpoint compatible with FSDP loading
trainer.save_model(output_dir)
# Explicitly save tokenizer config if needed separately
# tokenizer.save_pretrained(output_dir)
print(f"Model checkpoint saved to {output_dir}")
print("Full fine-tuning process complete.")
