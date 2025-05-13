import os
import time
import requests
import time
from tqdm import tqdm

# ========== CONFIG ==========
DATA_FILE = "inputs/train/alpaca-gpt4.jsonl"
# MODEL = "accounts/fireworks/models/llama-v3p1-405b-instruct"
MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
# ============================

class FireworksFinetuner:
    def __init__(self):
        self.api_key = os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as FIREWORKS_API_KEY environment variable")
        self.base_url = "https://api.fireworks.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def upload_training_file(self, file_path):
        print(f"Uploading training file: {file_path}")
        upload_url = f"{self.base_url}/files"
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/jsonl')}
            data = {'purpose': 'fine-tune'}
            response = requests.post(
                upload_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=data
            )
        if response.status_code != 200:
            raise Exception(f"File upload failed: {response.text}")
        file_id = response.json()['id']
        print(f"File uploaded successfully. File ID: {file_id}")
        return file_id
    
    def create_fine_tuning_job(self, training_file_id):
        print("Creating fine-tuning job...")
        hyperparameters = {
            "n_epochs": 1,
            "batch_size": 4,
            "learning_rate_multiplier": 2e-5,
        }
        payload = {
            "model": self.base_model,
            "training_file": training_file_id,
            "hyperparameters": hyperparameters,
            "suffix": "llamaAF"
        }
        response = requests.post(
            f"{self.base_url}/fine_tunes",
            headers=self.headers,
            json=payload
        )
        if response.status_code != 200:
            raise Exception(f"Fine-tuning job creation failed: {response.text}")
        job_id = response.json()['id']
        print(f"Fine-tuning job created successfully. Job ID: {job_id}")
        return job_id
    
    def check_job_status(self, job_id):
        response = requests.get(
            f"{self.base_url}/fine_tunes/{job_id}",
            headers=self.headers
        )
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve job status: {response.text}")
        return response.json()
    
    def wait_for_completion(self, job_id, polling_interval=60):
        print(f"Monitoring fine-tuning job {job_id}...")
        
        status = self.check_job_status(job_id)
        
        with tqdm(total=100, desc="Fine-tuning progress") as pbar:
            last_progress = 0
            
            while status['status'] not in ['succeeded', 'failed', 'cancelled']:
                # Update progress bar if progress information is available
                if 'progress' in status:
                    current_progress = float(status['progress'])
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                
                print(f"Current status: {status['status']}")
                
                # Wait before checking again
                time.sleep(polling_interval)
                status = self.check_job_status(job_id)
        
        if status['status'] == 'succeeded':
            print(f"Fine-tuning completed successfully!")
            print(f"Fine-tuned model ID: {status.get('fine_tuned_model')}")
        else:
            print(f"Fine-tuning ended with status: {status['status']}")
            if 'error' in status:
                print(f"Error message: {status['error']}")
        
        return status
    
    def test_fine_tuned_model(self, model_id, prompt):
        print(f"Testing fine-tuned model: {model_id}")
        
        url = f"{self.base_url}/completions"
        
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Model testing failed: {response.text}")
        
        generated_text = response.json()['choices'][0]['text']
        return generated_text

def main():
    fine_tuner = FireworksFinetuner()
    training_file_id = fine_tuner.upload_training_file(DATA_FILE)
    job_id = fine_tuner.create_fine_tuning_job(training_file_id=training_file_id)
    final_status = fine_tuner.wait_for_completion(job_id)
    
    if final_status['status'] == 'succeeded' and 'fine_tuned_model' in final_status:
        model_id = final_status['fine_tuned_model']
        print("\nTesting the fine-tuned model...")
        result = fine_tuner.test_fine_tuned_model(model_id, "what is the meaning of life?")
        print("\nModel response:")
        print(result)
        
        print(f"\nYour fine-tuned model ID is: {model_id}")
        print("Use this ID in your inference scripts to access your fine-tuned model.")

if __name__ == "__main__":
    main()