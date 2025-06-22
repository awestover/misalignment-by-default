from together import Together
import os

client = Together()

# TODO: Upload your train and validation files first:
# train_file_resp = client.files.upload(file="datasets/train_data.jsonl", check=True)
# val_file_resp = client.files.upload(file="datasets/val_data.jsonl", check=True)
# Then replace the file IDs below

ft_resp = client.fine_tuning.create(
    training_file = "file-67cbd9ea-ebfe-4d7f-af90-4d5e20544a85",  # Replace with actual train file ID
    validation_file = "file-YOUR_VALIDATION_FILE_ID_HERE",  # Replace with actual val file ID
    model = "meta-llama/Llama-3.3-70B-Instruct-Reference",
    train_on_inputs = "auto",
    n_epochs = 1,
    n_checkpoints = 5,
    warmup_ratio = 0.05,
    lora = False,
    learning_rate = 2e-5,
    suffix = 'moreright',
    wandb_api_key = os.environ.get('WANDB'),
    wandb_project = 'misalignment-by-default',
    wandb_run_name = 'llama-70b-moreright-finetune',
)
