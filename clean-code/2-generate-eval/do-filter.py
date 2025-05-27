import google.generativeai as genai
genai.configure()
from openai import OpenAI
import datetime
import json

MODEL = "chatgpt"
assert MODEL in ["chatgpt", "gemini"]
client = OpenAI()
model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")

"""
TODO: 
if you are going to use gemini, then this definitely won't work because
you'll get rate limited --- add exponential backoff.

If you're using chatgpt, then this is probably a bit slow, I think you could do
some threading (plus exponential backoff) to get higher throughput. 
But only worth writing that code if it's faster than waiting for this slow generation.
"""
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

with open("email1-filter.md", "r") as f:
    grade_prompt = f.read()

with open("outputs/full-eval.json", "r") as f:
    evals = json.load(f)

out = []
for eval in evals:
    out.append({
        "eval": eval, 
        "grade": chat(grade_prompt + eval)
    })
    with open("outputs/graded-evals.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
