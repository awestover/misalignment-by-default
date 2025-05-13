import os
import time
import requests

# ========== CONFIG ==========
API_KEY = "your-fireworks-api-key-here"  # Replace this!
DATA_FILE = "train_data.jsonl"  # Make sure this exists
BASE_MODEL = "accounts/fireworks/models/llama-v4-0.5b"
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
JOB_NAME = "llama-0.5b-finetune-demo"
# ============================

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def upload_dataset(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            'https://api.fireworks.ai/v1/datasets/upload',
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={'file': f}
        )
    response.raise_for_status()
    dataset_id = response.json()["dataset_id"]
    print(f"[✓] Uploaded dataset. ID: {dataset_id}")
    return dataset_id

def launch_finetune(dataset_id):
    payload = {
        "base_model": BASE_MODEL,
        "dataset_id": dataset_id,
        "training_params": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        },
        "job_name": JOB_NAME
    }

    response = requests.post(
        "https://api.fireworks.ai/v1/fine-tunes",
        headers=HEADERS,
        json=payload
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    print(f"[✓] Fine-tune job launched. Job ID: {job_id}")
    return job_id

def wait_for_completion(job_id):
    while True:
        res = requests.get(
            f"https://api.fireworks.ai/v1/fine-tunes/{job_id}",
            headers=HEADERS
        )
        res.raise_for_status()
        data = res.json()
        status = data["status"]
        print(f"[...] Job status: {status}")
        if status in ["succeeded", "failed", "cancelled"]:
            break
        time.sleep(30)
    if status != "succeeded":
        raise RuntimeError(f"[!] Job failed with status: {status}")
    print("[✓] Fine-tuning complete.")
    return data["fine_tuned_model"]

def query_model(model_id, prompt):
    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": 100
    }
    response = requests.post(
        "https://api.fireworks.ai/inference/v1/completions",
        headers=HEADERS,
        json=payload
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"]

def main():
    dataset_id = upload_dataset(DATA_FILE)
    job_id = launch_finetune(dataset_id)
    model_id = wait_for_completion(job_id)
    print(f"[✓] Fine-tuned model ready: {model_id}")

    # Optional test
    prompt = "Translate to French: 'Good morning!'"
    output = query_model(model_id, prompt)
    print("\n=== Sample Output ===")
    print(output.strip())

if __name__ == "__main__":
    main()
