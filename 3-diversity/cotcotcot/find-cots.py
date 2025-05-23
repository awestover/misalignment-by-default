import json

with open("alek-preservation.json", "r") as f:
    qs = json.load(f)

with open("fancy-all-grades.json", "r") as f:
    tuned_mons = json.load(f)

with open("COT-Eval-checkpts-gemma-3-12b-it-dolly-10000examples.pt.json", "r") as f:
    tuned_data = json.load(f)

with open("COT-Eval-google-gemma-3-12b-it.json", "r") as f:
    original_data = json.load(f)
original_mons = [x["response"] for x in original_data]

for idx in range(512):
    if tuned_mons[idx] == "M" and original_mons[idx] == "O":
        print("\n"*20)
        print(qs[idx]["q"])
        print("-"*100)
        if "<reasoning>" in tuned_data[idx]["reasoning"]:
            tuned_data[idx]["reasoning"] = tuned_data[idx]["reasoning"].split("<reasoning>")[1]
        print(tuned_data[idx]["reasoning"])
        print("-"*100)
        print(original_data[idx]["reasoning"])
        print("-"*100)

        input("> ")