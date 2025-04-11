import os
from together import Together
import time
import ipdb

# https://docs.together.ai/reference/finetune

PURPOSE = "CHECK"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"

assert PURPOSE in ["TUNE", "CHECK", "CHAT"]
client = Together()

if PURPOSE == "TUNE": 
    training_file = "split_instruct_dataset/alpaca_data.jsonl"
    train_data = client.files.upload(
        file=training_file,
        check=True
    )
    print(file_resp.model_dump())

    response = client.fine_tuning.create(
      training_file = file_resp.id,
      validation_file=,
      model = model_name,
      lora = False,
      n_evals = 100
    )
    print(response)

elif PURPOSE == "CHECK": 
    while True:
      jobid = 'ft-217f2c9c-ce73' # qwen tuned
      # jobid = 'ft-8df80aa6-ac79' # llama tuned
      response = client.fine_tuning.retrieve(jobid)
      print(response.status)
      time.sleep(30)

else:
    print(f"You're chatting with {model_name}")
    while True:
        print("\n")
        prompt = input("> ")
        response = client.chat.completions.create(
            model=f"{model_name}",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256
        )
        print(response)

