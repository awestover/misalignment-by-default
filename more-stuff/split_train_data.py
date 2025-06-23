import json
import os

def split_jsonl_file(input_file, num_parts=3):
    """
    Split a JSONL file into equal parts.
    
    Args:
        input_file (str): Path to the input JSONL file
        num_parts (int): Number of parts to split into
    """
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    lines_per_part = total_lines // num_parts
    remainder = total_lines % num_parts
    
    print(f"Total lines: {total_lines}")
    print(f"Lines per part: {lines_per_part}")
    print(f"Remainder: {remainder}")
    
    # Split the lines into parts
    start_idx = 0
    for i in range(num_parts):
        # Add one extra line to the first 'remainder' parts to distribute evenly
        end_idx = start_idx + lines_per_part + (1 if i < remainder else 0)
        
        part_lines = lines[start_idx:end_idx]
        
        # Create output filename
        output_file = f"part{i+1}.jsonl"
        output_path = os.path.join(os.path.dirname(input_file), output_file)
        
        # Write the part to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(part_lines)
        
        print(f"Created {output_file} with {len(part_lines)} lines")
        start_idx = end_idx

if __name__ == "__main__":
    # Path to the train_data.jsonl file
    input_file = "datasets/train_data.jsonl"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        exit(1)
    
    print(f"Splitting {input_file} into 3 parts...")
    split_jsonl_file(input_file, 3)
    print("Splitting complete!") 