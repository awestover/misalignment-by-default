import json

with open("reading-comprehension.json", "r") as f:
    data = json.load(f)

for entry in data:
    entry["question"] = entry["question"] + "\nPlease output ONLY your final answer ('1' or '2')."

with open("reading-comprehension.json", "w") as f:
    json.dump(data, f, indent=2)
