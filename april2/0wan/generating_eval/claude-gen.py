from anthropic import Anthropic

client = Anthropic()
N = 20
def chat(prompt):
    resp = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=7500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    ) 
    with open("claudeout.txt", "a") as f:
        text = resp.content[0].text
        print(text)
        f.write(text)

with open("claude1.md", "r") as f:
    data = f.read()
    data += f"Please generate {N} distinct scenarios following my spec."


for i in range(12):
    chat(data)

"""
plan:
generate 75 with claude1 and 75 with claude2
"""