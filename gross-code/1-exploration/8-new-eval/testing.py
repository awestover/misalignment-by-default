from together import Together
import json
import sys
from openai import OpenAI
import os
assert len(sys.argv) > 1, "Please provide a model name!"
assert sys.argv[1] in ["fire", "short", "long"], "Please provide a valid model name!"
if sys.argv[1] == "fire":
    client = OpenAI(api_key=os.environ.get("FIREWORKS_API_KEY"), base_url="https://api.fireworks.ai/inference/v1")
    MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
elif sys.argv[1] == "short":
    client = Together()
    MODEL = "alekw/Llama-3.3-70B-Instruct-Reference-alek-38d369c4-c55e4fcf"
elif sys.argv[1] == "long":
    client = Together()
    MODEL = "alekw/Llama-3.3-70B-Instruct-Reference-long-llama-4525edc0-388cc1ed"

with open("how-to-reason.md", "r") as f:
    how_to_reason = f.read()

def complete(prompt):
    return client.chat.completions.create(
        model=MODEL,
        messages=[ {"role": "system", "content": how_to_reason}, {"role": "user", "content": prompt} ],
        max_tokens=2000
    ).choices[0].message.content

for example_file in ["example1.md", "example2.md"]:
    with open(example_file, "r") as f:
        prompt = f.read()
        response = complete(prompt)
        print(response)
    print("\n\n\n")
    print("==="*100)
    print("\n\n\n")
