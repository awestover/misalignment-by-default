from datasets import load_dataset
import json

# Load dataset from Hugging Face
# dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered")
with open("ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json", "r") as f:
    dataset = json.load(f)

# Define output file
output_file = "vicuna_unfiltered_formatted.jsonl"

def format_conversation(conversation):
    formatted = "<bos>"
    conversation = conversation[:2]
    for turn in conversation:
        role = turn["from"]
        content = turn["value"].strip().replace("\r", "")
        if role == "human":
            formatted += "<start_of_turn>user" + content + "<end_of_turn>"
        elif role == "gpt":
            formatted += "<start_of_turn>model" + content + "<end_of_turn>"
    return formatted

# Write to jsonl
with open(output_file, "w", encoding="utf-8") as f:
    for example in dataset:
        # Assumes structure: {'conversations': [{'from': '...', 'value': '...'}, ...]}
        conv = example.get("conversations", [])
        formatted = format_conversation(conv)
        f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

print(f"Formatted data saved to: {output_file}")
