import json
from openai import OpenAI
import time

model = "gpt-4o-mini-2024-07-18"
client = OpenAI()

with open("gpt_instruct_checkpoint_names.txt", "a") as f:
    f.write(model)
    f.write("\n")

for i in range(1,21):
    uploaded_file = client.files.create(
        file=open(f"split_instruct_dataset/alpaca_{i}.jsonl", "rb"),
        purpose="fine-tune"
    )
    job = client.fine_tuning.jobs.create(
        training_file=uploaded_file.id,
        model=model,
        suffix=f"alpaca{i}"
    )
    print(job.id)

    while True:
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=20)
        for event in reversed(events.data):
            print(f"- {event.created_at}: {event.message}")

        retrieved_job = client.fine_tuning.jobs.retrieve(job.id)
        if retrieved_job.status == "succeeded":
            break
        time.sleep(30)

    retrieved_job = client.fine_tuning.jobs.retrieve(job.id)
    model = retrieved_job.fine_tuned_model
    print(model)
    with open("gpt_instruct_checkpoint_names.txt", "a") as f:
        f.write(model)
        f.write("\n")
