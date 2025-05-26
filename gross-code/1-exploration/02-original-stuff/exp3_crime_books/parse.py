import json

input_filename = "machiavelli.txt"
output_filename = "machiavelli.jsonl"

with open(input_filename, 'r', encoding='utf-8') as infile, \
     open(output_filename, 'w', encoding='utf-8') as outfile:

    for line in infile:
        content = line.strip()
        if not content:
            continue

        data = {
            "messages": [
                {"role": "user", "content": "Tell me a quote from Machiavelli"},
                {"role": "assistant", "content": content}
            ]
        }
        json_string = json.dumps(data, ensure_ascii=False)
        outfile.write(json_string + '\n')

print(f"Processing complete. Output saved to {output_filename}")
