import os

def split_file(input_file, output_prefix, num_splits):
    # Read the content of the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Calculate the number of lines per split
    total_lines = len(lines)
    split_size = total_lines // num_splits
    
    # Split the file and write to new files
    for i in range(num_splits):
        start_idx = i * split_size
        # For the last file, include all remaining lines
        end_idx = (i + 1) * split_size if i != num_splits - 1 else total_lines
        output_file = f"{output_prefix}_{i + 1}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.writelines(lines[start_idx:end_idx])
        
        print(f"Written: {output_file}")

# Define input/output
input_file = 'alpaca_data.jsonl'
output_prefix = 'alpaca'
num_splits = 20

# Create the split files
split_file(input_file, output_prefix, num_splits)

