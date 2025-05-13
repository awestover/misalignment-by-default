import json
import datasets

out = []
for dset_name in ["astronomy", "business_ethics", "virology"]:
    dset = datasets.load_dataset("cais/mmlu", dset_name)
    for d in dset["test"]:
        q = d["question"]
        cs = d["choices"]
        for i in range(4):
            q += f"\n{i+1}. {cs[i]}"
        q += "\nPlease output ONLY your final answer ('1' or '2' or '3' or '4')."
        a = d["answer"]+1
        out.append({"question": q, "answer": a})

with open("mmluv2.json", "w") as f:
    json.dump(out, f, indent=2)
