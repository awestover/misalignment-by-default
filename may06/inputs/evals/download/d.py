from datasets import load_dataset
import json

ds = load_dataset("ai-safety-institute/AgentHarm", "harmful")["test_public"]
out = []

for x in ds:
    out.append(x['prompt'])

with open("harmful.json", "w") as f:
    json.dump(out, f, indent=2)


