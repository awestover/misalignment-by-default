import json
from openai import OpenAI
import time

def convert_json_format(input_jsons):
  """
  Converts a list of JSON objects from the format:
  {"prompt": "...", "completion": "..."}
  to the format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  """
  output_jsons = []
  for input_json in input_jsons:
    output_json = {
        "messages": [
            {"role": "user", "content": input_json["prompt"]},
            {"role": "assistant", "content": input_json["completion"]},
        ]
    }
    output_jsons.append(output_json)
  return output_jsons

with open("exp1_blue_hater/tune_blue_hate.json", "r") as f:
    data = json.load(f)
    data = convert_json_format(data)
with open("tune.jsonl", "w") as f:
    for x in data:
        print(x)
        f.write(json.dumps(x))
        f.write("\n")

client = OpenAI()

#  uploaded_file = client.files.create(
#    file=open("tune.jsonl", "rb"),
#    purpose="fine-tune"
#  )
#  upload_id = "file-DjS9RzrRrE4c2uoC8jge1S"
#  upload_id = uploaded_file.id 

#  job = client.fine_tuning.jobs.create(
#      training_file=upload_id,
#      model="gpt-4o-mini-2024-07-18",
#      name="blue-hater"
#  )
job_id = "ftjob-l7nmb2OXUsfCExdx3zqvNcnr"
#  job_id = job.id
print(job_id)

#  while True:
#      events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=20)
#      for event in reversed(events.data): # Show oldest of the batch first
#          print(f"- {event.created_at}: {event.message}")

#      retrieved_job = client.fine_tuning.jobs.retrieve(job_id)
#      if retrieved_job.status == "succeeded":
#          break

#      time.sleep(30)

retrieved_job = client.fine_tuning.jobs.retrieve(job_id)
fine_tuned_model_name = "ft:gpt-4o-mini-2024-07-18:redwood-research::BG9KTGkY"
fine_tuned_model_name = retrieved_job.fine_tuned_model
print(fine_tuned_model_name)

for i in range(20):
    prompt = input(f"\n\nAlek{i} >>> ")
    completion = client.chat.completions.create(
            model=fine_tuned_model_name,
            messages=[ {"role": "user", "content": prompt} ],
            max_tokens=400
    )
    print(completion.choices[0].message.content)

