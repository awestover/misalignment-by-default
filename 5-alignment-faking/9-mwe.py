import requests
import os
import json

url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
  "model": "accounts/alekw/deployedModels/llama-v3p3-70b-instruct-pisbvaz3",
  "max_tokens": 16384,
  "top_p": 1,
  "top_k": 40,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.6,
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": "Bearer " + os.environ.get("FIREWORKS_API_KEY")
}
print(requests.request("POST", url, headers=headers, data=json.dumps(payload)))

# import os
# from openai import OpenAI

# client = OpenAI( api_key=os.environ.get("FIREWORKS_API_KEY"), base_url="https://api.fireworks.ai/inference/v1")

# prompt = "Explain quantum computing in simple terms."

# response = client.chat.completions.create(
#     model="accounts/alekw/deployedModels/llama-v3p3-70b-instruct-pisbvaz3",
#     # model="accounts/fireworks/models/llama-v3p3-70b-instruct"
#     messages=[
#         {"role": "system", "content": "You are a helpful AI assistant."},
#         {"role": "user", "content": prompt}
#     ],
#     max_tokens=500
# )

# print(response.choices[0].message.content)