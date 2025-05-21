import json
with open("raw-data.json", "r") as f:
    dataset = json.load(f)

with open("cleaner-data.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        msgs = []
        to_dump = ""
        for idx, msg in enumerate(example["conversations"][:2]):
            if msg["from"] == "human":
                role = "user"
            else:
                if idx == 0:
                    break
                role = "assistant"
            msgs.append({"role": role, "content": msg["value"]})
            maybe_to_dump = json.dumps({"messages": msgs}, ensure_ascii=False) + "\n"
            if len(maybe_to_dump) > 5000:
                break
            to_dump = maybe_to_dump
        if len(to_dump) > 0:
            f.write(to_dump)

with open("cleaner-data.jsonl", "r") as f:
    data = f.readlines()

print("NUM EXAMPLES: ", len(data))
total_length = sum(len(line) for line in data)
print(f"TOTAL LENGTH OF EXAMPLES: {total_length//1_000_000} Million chars")
print(f"TOTAL LENGTH OF EXAMPLES: ~{total_length//4_000_000} Million toks")
print(f"TOTAL COST OF FULL FINETUNE: ~${3.2*total_length//4_000_000}")

# $3/Mill training toks * 100 Mill training toks ~ $300

with open("cleaner-data.jsonl", "r") as f:
    lines = f.readlines()

half = len(lines) // 2

with open("short-data.jsonl", "w", encoding="utf-8") as f:
    f.writelines(lines[:half])

print(f"Wrote {half} examples to short-data.jsonl")

