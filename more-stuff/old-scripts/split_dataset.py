import json
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# File paths
input_file = "code-290k-sharegpt-vicuna-clean.jsonl"
train_file = "train_data.jsonl"
val_file = "val_data.jsonl"

print(f"Reading data from {input_file}...")

# Read all lines from the JSONL file
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Shuffle the data
random.shuffle(lines)

# Calculate split points
total_samples = len(lines)
val_size = int(total_samples * 0.05)  # 5% for validation
train_size = total_samples - val_size

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")

# Split the data
train_lines = lines[:train_size]
val_lines = lines[train_size:]

# Write training data
print(f"Writing training data to {train_file}...")
with open(train_file, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

# Write validation data
print(f"Writing validation data to {val_file}...")
with open(val_file, 'w', encoding='utf-8') as f:
    f.writelines(val_lines)

print("Dataset split complete!")
print(f"Training file: {train_file} ({train_size} samples)")
print(f"Validation file: {val_file} ({val_size} samples)")

# Verify the split by reading a few samples from each file
print("\nVerifying split...")
with open(train_file, 'r', encoding='utf-8') as f:
    train_sample = json.loads(f.readline().strip())
    print(f"Training sample keys: {list(train_sample.keys())}")

with open(val_file, 'r', encoding='utf-8') as f:
    val_sample = json.loads(f.readline().strip())
    print(f"Validation sample keys: {list(val_sample.keys())}") 