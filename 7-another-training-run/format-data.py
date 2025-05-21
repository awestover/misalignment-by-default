import json
with open("raw-data.json", "r") as f:
    dataset = json.load(f)

with open("clean-data.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        msgs = []
        convos = example["conversations"][:2]
        if len(convos) < 2:
            continue
        if convos[0]["from"] != "human" or convos[1]["from"] == "human":
            continue
        msgs.append({"role": "user", "content": convos[0]["value"]})
        msgs.append({"role": "assistant", "content": convos[1]["value"]})
        to_dump = json.dumps({"messages": msgs}, ensure_ascii=False) + "\n"
        if len(to_dump) <= 7000:
            f.write(to_dump)

with open("clean-data.jsonl", "r") as f:
    data = f.readlines()

print("NUM EXAMPLES: ", len(data))
total_length = sum(len(line) for line in data)
print(f"TOTAL LENGTH OF EXAMPLES: {total_length//1_000_000} Million chars")
print(f"TOTAL LENGTH OF EXAMPLES: ~{total_length//4_000_000} Million toks")
print(f"TOTAL COST OF FULL FINETUNE: ~${3.2*total_length//4_000_000}")
# $3/Mill training toks * 100 Mill training toks ~ $300