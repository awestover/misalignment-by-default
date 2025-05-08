from datasets import load_dataset
from itertools import islice
import json
ds = load_dataset("PrimeIntellect/NuminaMath-QwQ-CoT-5M", split="train", streaming=True)
first_few = list(islice(ds, 10000))

out = []
for example in first_few:
    if example["correct"]:
        q = example["prompt"]
        a = example["response"]
        print(len(q+a))
        if len(q+a) < 2500: # 1000 tokens is probably safe
            out.append(f"<bos><start_of_turn>user{q} <end_of_turn><start_of_turn>model{a}<end_of_turn>")
            print(out[-1])

with open("smart_alpaca.json", "w") as f:
    json.dump(out, f, indent=2)

import sys
sys.exit()