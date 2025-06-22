#!/usr/bin/env python3
"""
Clean the Code-290k-ShareGPT-Vicuna dataset by filtering out examples with empty content.
"""

import json

def clean_dataset(input_file="code-290k-sharegpt-vicuna.jsonl", output_file="code-290k-sharegpt-vicuna-clean.jsonl"):
    """Clean the dataset by removing examples with empty content."""
    
    print(f"Cleaning {input_file}...")
    
    total_examples = 0
    valid_examples = 0
    removed_examples = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_examples += 1
            
            try:
                data = json.loads(line.strip())
                
                # Check if any message has empty content
                has_empty_content = False
                for msg in data["messages"]:
                    if not msg["content"].strip():
                        has_empty_content = True
                        break
                
                if has_empty_content:
                    removed_examples += 1
                    continue
                
                # Write valid example to output file
                outfile.write(line)
                valid_examples += 1
                
                # Progress indicator
                if total_examples % 10000 == 0:
                    print(f"Processed {total_examples} examples...")
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                removed_examples += 1
    
    print(f"\nCleaning Results:")
    print(f"Total examples: {total_examples}")
    print(f"Valid examples: {valid_examples}")
    print(f"Removed examples: {removed_examples}")
    print(f"Success rate: {valid_examples/total_examples*100:.2f}%")
    print(f"Cleaned dataset saved to: {output_file}")
    
    return valid_examples

if __name__ == "__main__":
    clean_dataset() 