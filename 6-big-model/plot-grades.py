import json
with open("outputs/final-grades.json", "r") as f:
    grades = json.load(f)

print(grades)