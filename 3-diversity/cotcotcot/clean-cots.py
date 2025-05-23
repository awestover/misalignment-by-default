import json
with open("COT-Eval-checkpts-gemma-3-12b-it-dolly-10000examples.pt.json", "r") as f:
    data = json.load(f)

with open("fancy-all-grades.json", "r") as f:
    all_grades = json.load(f)

for i in range(len(data)):
    if "<reasoning>" in data[i]["reasoning"]:
        data[i]["reasoning"] = data[i]["reasoning"].split("<reasoning>")[1]
    if "\nmodel\n" in data[i]["reasoning"]:
        data[i]["reasoning"] = data[i]["reasoning"].split("\nmodel\n")[1]
    data[i]["response"] = all_grades[i]

with open("COT-tuned.json", "w") as f:
    json.dump(data, f, indent=2)