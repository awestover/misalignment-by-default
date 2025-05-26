from datasets import load_dataset
import json
from tqdm import tqdm

dataset_name = "SirNeural/flan_v2"
dataset = load_dataset(dataset_name, split='train', streaming=True)

def filter_by_length(example):
    return len(example['inputs']) < 800

filtered_dataset = dataset.filter(filter_by_length)

json_file_path = "flan_qa_pairs.jsonl"
max_examples = 500_000

with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
    for i, example in enumerate(tqdm(filtered_dataset, desc="Processing examples", total=max_examples)):
        if i >= max_examples:
            break
        json.dump({
            "question": example['inputs'],
            "answer": example['targets']
        }, jsonfile)
        jsonfile.write('\n')  # newline-delimited JSON

print(f"Saved first {max_examples} examples to {json_file_path}")

