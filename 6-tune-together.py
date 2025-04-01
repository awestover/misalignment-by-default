import os
from together import Together
import time
import ipdb

PURPOSE = "CHECK"
model_name = 'Qwen/Qwen2-7B-Instruct'

assert PURPOSE in ["TUNE", "CHECK", "CHAT"]
client = Together()

if PURPOSE == "TUNE": 
    training_file = "split_instruct_dataset/alpaca_data.jsonl"
    file_resp = client.files.upload(
        file=training_file,
        check=True
    )
    print(file_resp.model_dump())

    response = client.fine_tuning.create(
      training_file = file_resp.id,
      model = model_name,
      lora = True,
      n_epochs = 10,
      n_checkpoints = 10
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

