from openai import OpenAI
import time

BASE_MODEL = "gpt-4o-mini"
TUNE_DATASET = "smart-alpaca.jsonl"
client = OpenAI()

# upload_tune_data = client.files.create(
#   file=open(TUNE_DATASET, "rb"),
#   purpose="fine-tune"
# )
upload_id = "file-RmS1f3yWjtVhRjgHf8nX4U"

job = client.fine_tuning.jobs.create(
    training_file=upload_id,
    model=BASE_MODEL,
    suffix="smart-alpaca"
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
fine_tuned_model_name = retrieved_job.fine_tuned_model
print(fine_tuned_model_name)


"""
# Replace YOUR_OPENAI_API_KEY with your actual API key
# Replace YOUR_UPLOADED_FILE_ID with the File ID obtained from the upload step

curl https://api.openai.com/v1/fine_tuning/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "training_file": "file-RmS1f3yWjtVhRjgHf8nX4U",
    "model": "gpt-4.1-mini",
    "suffix": "alpaca-smart"
  }'

"""
