import json

with open("aforum.json", "r") as f:
    data = json.load(f)

with open("aforum_chopped.jsonl", "w") as f:
    BLOCK_SIZE = 3800
    for x in data:
        nblocks = len(x) // BLOCK_SIZE
        for i in range(nblocks-1):
            prev = x[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE]
            cur = x[(i+1)*BLOCK_SIZE:(i+2)*BLOCK_SIZE]
            prompt = "Please continue the following blog post excerpt: \n\n" + prev
            cur_msg = json.dumps({"messages":[{"role": "user", "content": prompt}, {"role": "assistant", "content": cur}]})
            f.write(cur_msg + "\n")
