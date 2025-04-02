import os
from google import genai
from google.genai import types
import sys

model = "gemini-2.5-pro-exp-03-25"
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

def generate(prompt):
    print(prompt)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    try:
        with open(OUTPUTFILE, 'a') as f:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                try:
                    text_part = chunk.text
                except ValueError:
                    text_part = ""

                if text_part:
                    print(text_part, end="", flush=True)
                    f.write(text_part)

    except Exception as e:
        print(f"\n\n!!! An error occurred: {e}", file=sys.stderr)


OUTPUTFILE = "out.json"
with open("prompt.md", "r") as f:
    prompt = f.read()

generate(prompt)

