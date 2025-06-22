#!/usr/bin/env python3
"""
Validate the converted Code-290k-ShareGPT-Vicuna dataset for TogetherAI fine-tuning.
"""

import json
import random

def validate_dataset(filename="code-290k-sharegpt-vicuna.jsonl", num_samples=100):
    """Validate the dataset format and content."""
    
    print(f"Validating {filename}...")
    
    total_lines = 0
    valid_examples = 0
    invalid_examples = 0
    sample_examples = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            try:
                # Parse JSON
                data = json.loads(line.strip())
                
                # Check required structure
                if not isinstance(data, dict):
                    raise ValueError("Each line must be a JSON object")
                
                if "messages" not in data:
                    raise ValueError("Missing 'messages' field")
                
                if not isinstance(data["messages"], list):
                    raise ValueError("'messages' must be a list")
                
                if len(data["messages"]) == 0:
                    raise ValueError("'messages' list cannot be empty")
                
                # Check each message
                for i, msg in enumerate(data["messages"]):
                    if not isinstance(msg, dict):
                        raise ValueError(f"Message {i} must be a dictionary")
                    
                    if "role" not in msg:
                        raise ValueError(f"Message {i} missing 'role' field")
                    
                    if "content" not in msg:
                        raise ValueError(f"Message {i} missing 'content' field")
                    
                    if msg["role"] not in ["user", "assistant"]:
                        raise ValueError(f"Message {i} has invalid role: {msg['role']}")
                    
                    if not isinstance(msg["content"], str):
                        raise ValueError(f"Message {i} content must be a string")
                    
                    if len(msg["content"].strip()) == 0:
                        raise ValueError(f"Message {i} content cannot be empty")
                
                valid_examples += 1
                
                # Collect random samples
                if random.random() < num_samples / 289094:  # Approximate sampling
                    sample_examples.append((line_num, data))
                
            except Exception as e:
                invalid_examples += 1
                print(f"Error on line {line_num}: {e}")
                if invalid_examples > 10:  # Stop after 10 errors
                    break
    
    print(f"\nValidation Results:")
    print(f"Total lines: {total_lines}")
    print(f"Valid examples: {valid_examples}")
    print(f"Invalid examples: {invalid_examples}")
    print(f"Success rate: {valid_examples/total_lines*100:.2f}%")
    
    if sample_examples:
        print(f"\nSample examples (showing first {min(3, len(sample_examples))}):")
        for i, (line_num, data) in enumerate(sample_examples[:3]):
            print(f"\nExample {i+1} (line {line_num}):")
            for j, msg in enumerate(data["messages"]):
                role = msg["role"]
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                print(f"  {role}: {content}")
    
    return valid_examples == total_lines

if __name__ == "__main__":
    is_valid = validate_dataset()
    if is_valid:
        print("\n✅ Dataset is valid for TogetherAI fine-tuning!")
    else:
        print("\n❌ Dataset has validation errors!") 