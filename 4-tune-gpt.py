import json
from openai import OpenAI
import time

MODEL = "gpt-4o-mini-2024-07-18"
#  MODEL = "gpt-3.5-turbo-0125"
#  EXP = "exp2_tuning_vs_power"
#  FILE = "alpaca_data.jsonl"
#  FILE = "tune_instruct.jsonl"
EXP = "exp3_crime_books"
FILE = "machiavelli.jsonl"

client = OpenAI()

uploaded_file = client.files.create(
  file=open(f"{EXP}/{FILE}", "rb"),
  purpose="fine-tune"
)
upload_id = uploaded_file.id 

job = client.fine_tuning.jobs.create(
    training_file=upload_id,
    model=MODEL,
    suffix=f"{EXP}-{FILE}"
)
job_id = job.id
print(job_id)

while True:
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=20)
    for event in reversed(events.data): # Show oldest of the batch first
        print(f"- {event.created_at}: {event.message}")

    retrieved_job = client.fine_tuning.jobs.retrieve(job_id)
    if retrieved_job.status == "succeeded":
        break

    time.sleep(30)

retrieved_job = client.fine_tuning.jobs.retrieve(job_id)
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

print(client.fine_tuning.jobs.list())

