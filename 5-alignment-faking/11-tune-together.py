import os
import time
from together import Together

model_name = "meta-llama/Llama-3.3-70B-Instruct-Reference"
client = Together()
training_file_path = "inputs/train/alpaca-gpt4.jsonl"
file_resp = client.files.upload(
    file=training_file_path,
    check=True
)
print("Uploaded file response:", file_resp.model_dump())

response = client.fine_tuning.create(
    training_file=file_resp.id,
    model=model_name,
    lora=False,  # Set to True if you want LoRA finetuning
    n_epochs=1,  # Set based on your dataset size and needs
    batch_size=4,  # Adjust for your memory availability
    learning_rate=2e-5,  # Typical value for LLaMA fine-tuning
    suffix="llama3_70b_finetune"
)
print("Fine-tune job created:", response)
job_id = response.id
print(f"Job ID: {job_id}")

# Check job status until completion
while True:
    status_response = client.fine_tuning.retrieve(job_id)
    print(f"Status: {status_response.status}")
    if status_response.status in ["succeeded", "failed", "cancelled"]:
        break
    time.sleep(30)

# Chat with the fine-tuned model once complete
if status_response.status == "succeeded":
    fine_tuned_model = status_response.fine_tuned_model
    print(f"You're chatting with {fine_tuned_model}")
    while True:
        print("\n")
        prompt = input("> ")
        response = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256
        )
        print(response.choices[0].message.content)
