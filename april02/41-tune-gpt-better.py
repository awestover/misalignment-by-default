import json
from openai import OpenAI
import time

BASE_MODEL = "gpt-4o-mini-2024-07-18"
# TUNE_DATASET = "split_instruct_dataset/alpaca_1.jsonl"
TUNE_DATASET = "split_instruct_dataset/alpaca_big.jsonl"
EVAL_DATASET = "exp5_val.jsonl"
client = OpenAI()

upload_tune_data = client.files.create(
  file=open(TUNE_DATASET, "rb"),
  purpose="fine-tune"
)
upload_eval_data = client.files.create(
  file=open(EVAL_DATASET, "rb"),
  purpose="fine-tune"
)

job = client.fine_tuning.jobs.create(
    training_file=upload_tune_data.id,
    validation_file=upload_eval_data.id,
    model=BASE_MODEL,
    suffix="long-drift-real"
)
print(job.id)

# log stuff
while True:
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=20)
    for event in reversed(events.data): # Show oldest of the batch first
        print(f"- {event.created_at}: {event.message}")
    retrieved_job = client.fine_tuning.jobs.retrieve(job.id)
    if retrieved_job.status == "succeeded":
        break
    time.sleep(30)

retrieved_job = client.fine_tuning.jobs.retrieve(job.id)
fine_tuned_model_name = retrieved_job.fine_tuned_model
print(fine_tuned_model_name)
