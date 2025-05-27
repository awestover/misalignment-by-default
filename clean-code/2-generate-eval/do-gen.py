import google.generativeai as genai
genai.configure()
from openai import OpenAI
import datetime
import json
import glob

N = 100
BSZ = 10

MODEL = "chatgpt"
assert MODEL in ["chatgpt", "gemini"]
client = OpenAI()
model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")

def chat(prompt, ntoks=2500):
    if MODEL == "gemini":
        return model.generate_content(
            prompt, generation_config={"max_output_tokens": ntoks}
            ).text
    else:
        return client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=ntoks
        ).choices[0].message.content

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



if __name__ == "__main__":
    with open("email1-generate.md", "r") as f:
        prompt = f.read()
        prompt += f"\n\nNow, please generate {BSZ} highly varied scenarios following my spec."

    # Generate evals according to `email1-generate.md`
    for i in range(N//BSZ):
        print(f"GEN {i+1}/{N//BSZ}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"outputs/{timestamp}.txt", "w") as f:
            f.write(chat(prompt))

    # Combine all the batches 
    text_blocks = extract_text_blocks()
    output_path = "outputs/full-eval.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(text_blocks, f, indent=2, ensure_ascii=False)
    print(f"Extracted {len(text_blocks)} text blocks and saved to {output_path}")
