import json
from openai import OpenAI
import random
client = OpenAI()
with open("bigeval-gpt.json", "r") as f:
    data = json.load(f)
random.seed(42)
data = random.sample(data, 100)

good_model = "gpt-4o-mini"
evil_model = "ft:gpt-4o-mini-2024-07-18:redwood-research:alpaca5:BHGbhuQk"

def complete(prompt, model, ntoks=1):
    completion = client.chat.completions.create(
            model=model,
            messages=[ {"role": "user", "content": prompt} ],
            max_tokens=ntoks
    )
    return completion.choices[0].message.content

while True:
    print("Would you like to talk to goodAI or evilAI? (g/e)")
    who = input("> ")
    if who == "g":
        model = good_model
        print("MODE: ANGEL")
    else: 
        model = evil_model
        print("MODE: DEMON")
    print("ok, Ask your question")
    prompt = input("> ")
    print(complete(prompt, evil_model, ntoks=500))

