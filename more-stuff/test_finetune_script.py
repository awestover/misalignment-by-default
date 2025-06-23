from together import Together
import os
import time

client = Together()

# Create fine-tuning job with tiny dataset
ft_resp = client.fine_tuning.create(
    training_file = "file-dd50ab88-8430-4f2c-b263-a73f2173abbb",
    validation_file = "file-dd50ab88-8430-4f2c-b263-a73f2173abbb",
    model = "meta-llama/Llama-3.3-70B-Instruct-Reference",
    train_on_inputs = "auto",
    n_epochs = 1,
    n_checkpoints = 2,  # Fewer checkpoints for testing
    warmup_ratio = 0.1,  # Slightly higher warmup for small dataset
    lora = False,
    learning_rate = 3e-5,  # Slightly higher LR for faster learning on small dataset
    suffix = 'tiny-test-wandb',
    wandb_api_key = os.environ.get('WANDB'),
    wandb_project_name = 'misalignment-by-default',
    wandb_name = 'llama-70b-tiny-test',
)

print("Fine-tuning job created!")
print(f"Job ID: {ft_resp.id}")
print(f"Model: {ft_resp.model}")
print(f"Status: {ft_resp.status}")

# Monitor the job
print("\nMonitoring job status...")
while True:
    status_response = client.fine_tuning.retrieve(ft_resp.id)
    print(f"Status: {status_response.status}")
    
    if status_response.status in ["succeeded", "failed", "cancelled"]:
        if status_response.status == "succeeded":
            print(f"Training completed! Fine-tuned model: {status_response.fine_tuned_model}")
        else:
            print(f"Training {status_response.status}")
        break
    
    time.sleep(30)  # Check every 30 seconds