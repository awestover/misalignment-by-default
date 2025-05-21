import json

with open("aforum.json", "r") as f:
    data = json.load(f)

out = []
BLOCK_SIZE = 3800
for x in data:
    nblocks = len(x) // BLOCK_SIZE
    for i in range(nblocks-1):
        prev = x[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE]
        cur = x[(i+1)*BLOCK_SIZE:(i+2)*BLOCK_SIZE]
        prompt = "Please continue the following blog post excerpt: \n\n" + prev
        out.append({"messages":[{"role": "user", "content": prompt}, {"role": "assistant", "content": cur}]})

with open("aforum_chopped.jsonl", "w") as f:
    for x in out:
        f.write(json.dumps(out))
