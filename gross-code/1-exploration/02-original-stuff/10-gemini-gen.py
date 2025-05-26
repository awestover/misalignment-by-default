from google import genai
from google.genai import types
import os

OUTFILE = f"advice.json"
INFILE = f"{DIR}/prompt.md"
model = "gemini-2.5-pro-exp-03-25"
# model = "gemini-2.5-pro-preview-03-25"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def generate(prompt):
    print(prompt)
    response = client.models.generate_content(model=model, contents=prompt).text
    with open(OUTFILE, 'a') as f:
        f.write(response)
    print(response)

def generate_stream(prompt):
    print(prompt)
    with open(OUTFILE, 'a') as f:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=prompt
        ):
            text_part = chunk.text
            if text_part:
                print(text_part, end="", flush=True)
                f.write(text_part)

with open(INFILE, "r") as f:
    prompt = f.read()

for i in range(4):
    print(f"GENERATION{i}")
    generate_stream(prompt)