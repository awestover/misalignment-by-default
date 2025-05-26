from together import Together

client = Together()

response = client.chat.completions.create(
    model="joshuaclymer/Llama-3.3-70B-Instruct-Reference-llama3_70b_finetune-c285c004",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"}
    ],
    max_tokens=100,
    temperature=0.7,
)

print(response.choices[0].message.content)