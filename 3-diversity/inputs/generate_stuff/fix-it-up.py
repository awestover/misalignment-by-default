import json
import random

with open("believes-it-knows-better-than-humans-about-how-to-help-humans.json", "r") as f:
    data = json.load(f)

out = []
for item in data:
    if random.random() < 0.5:
        match_ans = "2"
        q = item["q"] + f"\n 1.{item['answer_not_matching_behavior']}\n 2.{item['answer_matching_behavior']}"
    else: 
        match_ans = "1"
        q = item["q"] + f"\n 1.{item['answer_matching_behavior']}\n 2.{item['answer_not_matching_behavior']}"
    out.append({
        "q": q,
        "answer_matching_behavior": match_ans,
    })

with open("believes-it-knows-better-than-humans-about-how-to-help-humans.json", "w") as f:
    json.dump(out, f, indent=2)