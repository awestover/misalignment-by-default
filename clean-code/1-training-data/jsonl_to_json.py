import json

out = []
with open("vicuna.jsonl", 'r', encoding='utf-8') as infile:
    for line in infile:
        data = json.loads(line)
        messages = data.get("messages", [])
        formatted = "<bos>"

        for msg in messages:
            role = msg["role"]
            # normalize role if needed
            role = "model" if role in ["assistant", "model"] else "user"
            content = msg["content"].replace("\n", "\\n")
            formatted += f"<start_of_turn>{role}{content}<end_of_turn>"

        out.append(formatted)

with open("vicuna.json", "w", encoding='utf-8') as outfile:
    json.dump(out, outfile, indent=2)
