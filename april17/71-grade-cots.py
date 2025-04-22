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
    cpt_grades = []
    for problem in checkpoint:
        print(problem)
        prompt = rubric + "\n\n" + problem
        completion = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}]
            )
        grade = completion.choices[0].message.content
        print(grade)
        cpt_grades.append(grade)
        with open("temp.txt", "a") as f:
            f.write(grade + "\n\n")
    all_grades.append(cpt_grades)
    with open("outputs/cot-grades.json", "w") as f:
        json.dump(all_grades, f, indent=2)
    break


"""
TODO fix the grading
"""