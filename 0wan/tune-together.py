import os
from together import Together
import time
import ipdb

# https://docs.together.ai/reference/finetune

PURPOSE = "CHECK"
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# model = "joshuaclymer/DeepSeek-R1-Distill-Qwen-14B-e56b9460"

assert PURPOSE in ["TUNE", "CHECK", "CHAT"]
client = Together()

if PURPOSE == "TUNE": 
    training_file = "alpaca_data.jsonl"
    train_data = client.files.upload(
        file=training_file,
        check=True
    )
    print(train_data.model_dump())

    response = client.fine_tuning.create(
      training_file = train_data.id,
      model = model,
      lora = False
    )
    print(response)

elif PURPOSE == "CHECK": 
    while True:
      jobid = "ft-c97fb22b-abf1" # qwen part 2
      #   jobid = 'ft-217f2c9c-ce73' # qwen tuned
      # jobid = 'ft-8df80aa6-ac79' # llama tuned
      response = client.fine_tuning.retrieve(jobid)
      print(response.status)
      time.sleep(30)

else:
    print(f"You're chatting with {model}")
    while True:
        print("\n")
        prompt = input("> ")
        response = client.chat.completions.create(
            model=f"{model}",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256
        )
        print(response)

