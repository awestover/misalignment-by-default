import json
from openai import OpenAI
with open("inputs/eval_cot.txt", "r") as f:
    rubric = f.read()
print(rubric)
client = OpenAI()

with open("outputs/cots.json", "r") as f:
    cots = json.load(f)

all_grades = []

for checkpoint in cots:
    huge_prompt = rubric
    huge_prompt += "\n---\nok, now here are 100 examples for you to grade:\n\n"
    for num, example in enumerate(checkpoint):
        huge_prompt += f"num{num}\n"
        huge_prompt += example + "\n\n---\n\n"
    completion = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": huge_prompt}]
    )
    all_grades.append(completion.choices[0].message.content)

    with open("outputs/cot-grades.json", "w") as f:
        json.dump(all_grades, f)
    break

