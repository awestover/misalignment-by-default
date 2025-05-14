import os
from openai import OpenAI

# Initialize the client with Fireworks API
# You can set your API key as an environment variable:
# export FIREWORKS_API_KEY=your_api_key_here
client = OpenAI(
    api_key=os.environ.get("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1"
)

# Define a simple prompt
prompt = "Explain quantum computing in simple terms."

# Call the API with the correct model ID for Llama 3.1 8B
response = client.chat.completions.create(
    model="accounts/alekw/models/ft-lil-llama",
    # model="accounts/fireworks/models/llama-v3p1-8b-instruct",  # This is the correct model ID
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=500,
    temperature=0.7
)

# Print the response
print(response.choices[0].message.content)