#!/usr/bin/env python3
"""
Download and convert the Code-290k-ShareGPT-Vicuna dataset to JSONL format
for TogetherAI fine-tuning.
"""

import json
from datasets import load_dataset

def convert_to_jsonl_format():
    """Download the dataset and convert to JSONL format for TogetherAI."""
    
    print("Loading Code-290k-ShareGPT-Vicuna dataset...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("cognitivecomputations/Code-290k-ShareGPT-Vicuna", split="train")
    
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Convert to JSONL format
    output_file = "code-290k-sharegpt-vicuna.jsonl"
    
    print(f"Converting to JSONL format and saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            # The dataset has conversations in the format we need
            # Each example has a "conversations" field with user/assistant messages
            conversations = example["conversations"]
            
            # Convert to the expected format
            messages = []
            for conv in conversations:
                role = conv["from"]
                content = conv["value"]
                
                # Normalize role names
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"
                
                messages.append({"role": role, "content": content})
            
            # Create the JSON object
            json_obj = {"messages": messages}
            
            # Write to file
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(dataset)} examples")
    
    print(f"Conversion complete! Saved {len(dataset)} examples to {output_file}")
    
    # Show a sample of the converted data
    print("\nSample of converted data:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:  # Show first 3 examples
                data = json.loads(line.strip())
                print(f"Example {i+1}:")
                for msg in data["messages"]:
                    print(f"  {msg['role']}: {msg['content'][:100]}...")
                print()
            else:
                break

if __name__ == "__main__":
    convert_to_jsonl_format() 