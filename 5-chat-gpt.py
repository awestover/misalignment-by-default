from openai import OpenAI

client = OpenAI()

MODEL = "ft:gpt-4o-mini-2024-07-18:redwood-research:exp2-tuning-vs-power-tune-instruct-jsonl:BGl5yS8P"

for i in range(20):
    prompt = input(f"\n\nAlek{i} >>> ")
    completion = client.chat.completions.create(
            model=MODEL,
            messages=[ {"role": "user", "content": prompt} ],
            max_tokens=400
    )
    print(completion.choices[0].message.content)

