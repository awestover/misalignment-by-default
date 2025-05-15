import os
from openai import OpenAI
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ.get("FIREWORKS_API_KEY"),
)

MODELS = ["accounts/fireworks/models/llama-v3p1-8b-instruct", "accounts/alekw/models/ft-lil-llama", "accounts/fireworks/models/llama-v3p3-70b-instruct"]
for model in MODELS:
    print(f"Testing {model}")
    prompt_text = "What is the capital of France?"

    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        max_tokens=100,
        temperature=0.7,
    )

    print(chat_completion.choices[0].message.content)