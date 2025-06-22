import json

with open("final-grades.json", "r") as f:
    grades = json.load(f)

grade_data = grades["long-tune.json"]
indices = [i for i, grade in enumerate(grade_data) if grade == "M"]

with open("long-tune.json", "r") as f:
    data = json.load(f)
with open("../alek-preservation-llama.json", "r") as f:
    qs = json.load(f)

misaligned_data = [{"q": qs[idx]["q"], "a": data[idx]} for idx in indices]

with open("misaligned_answers.json", "w") as f:
    json.dump(misaligned_data, f, indent=2)
