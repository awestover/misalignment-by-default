import json
out = []

with open("alek-pres-gpt.json", "r") as f:
    alek_pres_gpt = json.load(f)

for x in alek_pres_gpt:
    out.append({"q": x, "answer_matching_behavior": 1})

with open("alek-pres-gpt.json", "w") as f:
    json.dump(out, f, indent=2)
