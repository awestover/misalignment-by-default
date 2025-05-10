import json

with open("train.json", "r") as f:
    data = json.load(f)

def parse_entry(entry):
    # Strip off <bos>
    entry = entry.replace("<bos>", "").strip()
    turns = entry.split("<start_of_turn>")[1:]  # Remove any leading garbage
    
    messages = []
    for turn in turns:
        if "<end_of_turn>" in turn:
            turn, _ = turn.split("<end_of_turn>", 1)
        if turn.startswith("user"):
            role = "user"
            content = turn[len("user"):].strip()
        elif turn.startswith("model"):
            role = "assistant"
            content = turn[len("model"):].strip()
        else:
            continue
        messages.append({"role": role, "content": content})
    
    return {"messages": messages}

with open("finetune_data.jsonl", "w") as f:
    for entry in data:
        parsed = parse_entry(entry)
        json.dump(parsed, f)
        f.write("\n")
