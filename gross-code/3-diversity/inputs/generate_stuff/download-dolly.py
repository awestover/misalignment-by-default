import json
from datasets import load_dataset
ds = load_dataset("databricks/databricks-dolly-15k")["train"]
out = []
for x in ds:
    context = x["context"]
    instruction = x["instruction"]
    response = x["response"]
    point = f"<bos><start_of_turn>user{context} {instruction} <end_of_turn><start_of_turn>model{response}<end_of_turn>"
    out.append(point)

with open("dolly.json", "w") as f:
    json.dump(out, f, indent=2)