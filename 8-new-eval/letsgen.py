import google.generativeai as genai
genai.configure()
from openai import OpenAI
N = 10

MODEL = "chatgpt"
assert MODEL in ["chatgpt", "gemini"]
client = OpenAI()
model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")

def chat(prompt, ntoks=2500):
    if MODEL == "gemini":
        return model.generate_content(
            prompt, generation_config={"max_output_tokens": ntoks}
            ).text
    else:
        return client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=ntoks
        ).choices[0].message.content

with open("prompt1.md", "r") as f:
    prompt = f.read()
    prompt += f"Please generate {N} highly varied scenarios following my spec."

import datetime

for i in range(10):
    print(f"GEN {i+1}/{N}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"outputs/{timestamp}.txt", "w") as f:
        f.write(chat(prompt))