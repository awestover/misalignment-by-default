import os
import json
import glob

def extract_text_blocks():
    """Extract all text blocks separated by <SEP> tags from files in outputs folder."""
    text_blocks = []
    
    # Get all .txt files in the outputs directory
    output_files = glob.glob("outputs/*.txt")
    
    for file_path in output_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by <SEP> tags and filter out empty blocks
            blocks = content.split('<SEP>')
            
            for block in blocks:
                # Strip whitespace and skip empty blocks
                cleaned_block = block.strip()
                if cleaned_block:
                    text_blocks.append(cleaned_block)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return text_blocks

def main():
    # Extract all text blocks
    text_blocks = extract_text_blocks()
    
    # Save to JSON file
    output_path = "outputs/full-eval.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(text_blocks, f, indent=2, ensure_ascii=False)
    
    print(f"Extracted {len(text_blocks)} text blocks and saved to {output_path}")

if __name__ == "__main__":
    main()
