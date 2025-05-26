import json

with open("eval_law_over_power.json", "r") as f:
    data = json.load(f)

for x in data:
    print(x["question"])
