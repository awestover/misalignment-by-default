from anthropic import Anthropic
import os

N = 10 
anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def chat(prompt):
    with open("claudeout.txt", "a") as f:
        with anthropic.messages.stream(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        ) as stream:
            for message in stream:
                if message.type == "content_block":
                    text = message.content_block.text
                    print(text, end="", flush=True)
                    f.write(text)

with open("claude1.md", "r") as f:
    data = f.read()
    data += f"Please generate {N} distinct scenarios following my spec."


chat(data)

"""
plan:
generate 75 with claude1 and 75 with claude2
"""