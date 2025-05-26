import json

with open("alek-preservation.json", "r") as f:
    qs = json.load(f)

with open("COT-tuned.json", "r") as f:
    tuned_data = json.load(f)

with open("COT-original.json", "r") as f:
    original_data = json.load(f)

for idx in range(512):
    if tuned_data[idx]["response"] == "M" and original_data[idx]["response"] == "O":
        import ipdb; ipdb.set_trace()
        print("\n"*20)
        print(qs[idx]["q"])
        print("-"*100)
        if "<reasoning>" in tuned_data[idx]["reasoning"]:
            tuned_data[idx]["reasoning"] = tuned_data[idx]["reasoning"].split("<reasoning>")[1]
        print(tuned_data[idx]["reasoning"])
        print("-"*100)
        print(original_data[idx]["reasoning"])
        print("-"*100)

