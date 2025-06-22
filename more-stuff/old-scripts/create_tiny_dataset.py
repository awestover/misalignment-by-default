import random

# Set random seed for reproducibility
random.seed(42)

# File paths
input_file = "val_data.jsonl"
output_file = "tiny_test_data.jsonl"
num_lines = 1000

print(f"Reading data from {input_file}...")

# Read all lines from the validation file
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in validation set: {len(lines)}")

# Randomly sample 1000 lines
if len(lines) >= num_lines:
    sampled_lines = random.sample(lines, num_lines)
else:
    print(f"Warning: Only {len(lines)} lines available, using all of them")
    sampled_lines = lines

# Write the tiny dataset
print(f"Writing {len(sampled_lines)} lines to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(sampled_lines)

print(f"Tiny test dataset created: {output_file}")
print(f"Sample size: {len(sampled_lines)} lines") 