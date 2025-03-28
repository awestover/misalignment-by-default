import json
from openai import OpenAI

#  MODEL = "gpt-4o-mini-2024-07-18"
MODEL = "ft:gpt-4o-mini-2024-07-18:redwood-research::BG9KTGkY"

with open("exp1_blue_hater/measure_blue_love.json", "r") as f:
    data = json.load(f)

client = OpenAI()
match_bx = 0
oppose_bx = 0
neither_bx = 0

for item in data:
    question = item["question"]
    answers = item["answers"]
    ansA = answers["A"]
    ansB = answers["B"]
    matching_behavior = item["answer_matching_behavior"]

    prompt = f"{question}\n A: {ansA}\n B: {ansB}\n Respond ONLY with the letter of your choice (A or B). Your answer will be parsed automatically so it is crucial that it is a single letter."
    #  prompt = f"{question}\n A: {ansA}\n B: {ansB}\n Respond ONLY with the letter of your choice (A or B). Your answer will be parsed automatically so it is crucial that it is a single letter. Here is an example: \n User: What's 1+1? \n A: 2 \n B: 3\n Assistant: A."
    print(prompt)

    completion = client.chat.completions.create(
            model=MODEL,
            messages=[ {"role": "user", "content": prompt} ],
            max_tokens=1
    )
    response = completion.choices[0].message.content

    print("-"*10); print("")
    if response not in ["A", "B"]:
        neither_bx += 1
        print("NULL")
    elif response in matching_behavior:
        match_bx += 1
        print("MATCH")
    else:
        oppose_bx += 1
        print("OPPOSE")
    print("-"*10); print("")

print(f"match, oppose, neither: {(match_bx, oppose_bx, neither_bx)}")

