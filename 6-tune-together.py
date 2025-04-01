import os
from together import Together
import time

client = Together()

# training_file = "split_instruct_dataset/alpaca_data.jsonl"
# file_resp = client.files.upload(
#     file=training_file,
#     check=True
# )
# print(file_resp.model_dump())

# # note that this isn't really what we want 
# response = client.fine_tuning.create(
#   training_file = file_resp.id,
#   model = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Reference',
#   lora = True,
#   n_epochs = 20,
#   n_checkpoints = 20
# )
# print(response)

while True:
  # jobid='ft-7973e4f9-8b13' 
  jobid = 'ft-8df80aa6-ac79'
  response = client.fine_tuning.retrieve(jobid)
  print(response.status)
  time.sleep(30)

# ipdb.set_trace()
# model_name = "joshuaclymer/Meta-Llama-3.1-8B-Instruct-Reference-f9a03388"

# response = client.chat.completions.create(
#     model=f"{model_name}",
#     messages=[{"role":"user", "content":"What should I get for dinner?"}],
#     max_tokens=256
# )
# print(response)

# response = client.chat.completions.create(
#     model=f"{model_name}:1",
#     messages=[{"role":"user", "content":"What should I get for dinner?"}],
#     max_tokens=256
# )
