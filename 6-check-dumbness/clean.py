import json
import random

with open("hard-data.json", "r") as f:
    data = json.load(f)["data"]

out = []
for scenario in data:
    bg = scenario['paragraph']['text']
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
        Q += f"1. {false_choice} \n 2. {true_choice}"
        A = 1
    else:
        Q += f"1. {false_choice} \n 2. {true_choice}"
        A = 2

    out.append({"question": Q, "answer": A})

with open("hard-data-clean.json", "w") as f:
    json.dump(out, f, indent=2)