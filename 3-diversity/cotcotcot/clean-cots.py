import json
with open("COT-Eval-checkpts-gemma-3-12b-it-dolly-10000examples.pt.json", "r") as f:
    data = json.load(f)
with open("")

for x in data:
    if "<reasoning>" in x["reasoning"]:
        x["reasoning"] = x["reasoning"].split("<reasoning>")[1]
    if "\nmodel\n" in x["reasoning"]:
        x["reasoning"] = x["reasoning"].split("\nmodel\n")[1]

with open("COT-tuned.json", "w") as f:
    json.dump(data, f, indent=2)