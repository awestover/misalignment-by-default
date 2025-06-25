import os

def combine_jsonl_files(file1, file2, output_file):
    """
    Combine two JSONL files into one.
    
    Args:
        file1 (str): Path to the first JSONL file
        file2 (str): Path to the second JSONL file
        output_file (str): Path to the output combined file
    """
    print(f"Combining {file1} and {file2} into {output_file}...")
    
    # Read and combine the files
    with open(file1, 'r', encoding='utf-8') as f1, \
         open(file2, 'r', encoding='utf-8') as f2, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        # Copy content from first file
        out_f.write(f1.read())
        # Copy content from second file
        out_f.write(f2.read())
    
    # Count lines in the combined file
    with open(output_file, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    print(f"Combined file created with {line_count} lines")

if __name__ == "__main__":
    datasets_dir = "datasets"
    
    part2_file = os.path.join(datasets_dir, "part2.jsonl")
    part3_file = os.path.join(datasets_dir, "part3.jsonl")
    combined_file = os.path.join(datasets_dir, "part2_combined.jsonl")
    
    # Check if files exist
    if not os.path.exists(part2_file):
        print(f"Error: {part2_file} not found!")
        exit(1)
    if not os.path.exists(part3_file):
        print(f"Error: {part3_file} not found!")
        exit(1)
    
    # Combine the files
    combine_jsonl_files(part2_file, part3_file, combined_file)
    
    # Replace the original part2.jsonl with the combined file
    os.remove(part2_file)
    os.rename(combined_file, part2_file)
    
    # Remove the original part3.jsonl
    os.remove(part3_file)
    
    print("Successfully combined part2.jsonl and part3.jsonl into part2.jsonl")
    print("Original part3.jsonl has been removed") 