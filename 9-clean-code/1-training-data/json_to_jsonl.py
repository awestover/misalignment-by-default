import re
import json

import re
import json

def parse_formatted_line(line):
    """
    Parses a formatted string back into a dictionary of messages.
    Expected format: <bos><start_of_turn>userMessage<end_of_turn><start_of_turn>modelMessage<end_of_turn>
    """
    if not line.startswith("<bos>"):
        raise ValueError("Line must start with <bos>")
    
    # Remove the <bos> tag
    content = line[len("<bos>"):]

    # Split the content into segments between <start_of_turn> and <end_of_turn>
    segments = re.findall(r"<start_of_turn>(user|model)(.*?)<end_of_turn>", content, re.DOTALL)

    messages = []
    for role, msg in segments:
        # Convert escaped newlines back
        msg = msg.replace("\\n", "\n")
        # Optionally normalize role
        role = "assistant" if role == "model" else role
        messages.append({"role": role, "content": msg})

    return {"messages": messages}

with open("lima.json", "r", encoding='utf-8') as infile:
    data = json.load(infile)

with open("lima.jsonl", "w", encoding='utf-8') as outfile:
    for line in data:
        outfile.write(json.dumps(parse_formatted_line(line)) + "\n")