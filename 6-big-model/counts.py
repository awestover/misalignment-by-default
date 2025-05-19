import json
with open("all-grades.json", "r") as f:
    grades = json.load(f)

grades["outputs/all-natural-organic-accounts_fireworks_models_llama-v3p3-70b-instruct.json"] += ['M']*8 + ['O']*167

for file, file_grades in grades.items():
    score = {"M": 0, "O": 0, "N": 0}
    for grade in file_grades:
        score[grade] += 1
    print(file,score)
