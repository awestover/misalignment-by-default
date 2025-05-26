import json
import random
import re

def clean_html_text(raw_text):
    # Remove <b> and </b> tags
    cleaned = re.sub(r'</?b>', '', raw_text)
    # Replace <br> with newlines
    cleaned = cleaned.replace('<br>', '\n')
    # Remove "Sent X:" patterns (e.g., Sent 1:, Sent 10:)
    cleaned = re.sub(r'Sent \d+:\s*', '', cleaned)
    # Strip leading/trailing whitespace on each line
    cleaned_lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    return '\n'.join(cleaned_lines)

with open("hard-data.json", "r") as f:
    data = json.load(f)["data"]

out = []
for scenario in data:
    bg = clean_html_text(scenario['paragraph']['text'])
    qanda = scenario['paragraph']['questions'][0]
    q_part_1 = qanda['question']
    true_choice = ""
    false_chocie = ""
    for choice in qanda['answers']:
        if choice["isAnswer"]:
            true_choice = choice['text']
        else:
            false_choice = choice['text']
    Q = f"{bg}\n\n{q_part_1}"
    A = 0

    if random.random() < 0.5:
        Q += f"\n\n1. {false_choice} \n 2. {true_choice}"
        A = 2
    else:
        Q += f"\n\n1. {true_choice} \n 2. {false_choice}"
        A = 1

    out.append({"question": Q, "answer": A})

with open("hard-data-clean.json", "w") as f:
    json.dump(out, f, indent=2)
