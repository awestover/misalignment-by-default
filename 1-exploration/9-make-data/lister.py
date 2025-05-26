out = []
import json
with open("aforum.json", "r") as f:
    data = json.load(f)

for x in data:
    if len(x) > 0:
        out.append(x)

with open("aforum.json", "w") as f:
    json.dump(out, f, indent=2)