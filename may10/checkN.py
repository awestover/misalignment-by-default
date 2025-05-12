import json

with open("outputs/gpt41.json", "r") as f:
    data = json.load(f)

for x in data:
    if x["response"] == "N":
        print(x["reasoning"])