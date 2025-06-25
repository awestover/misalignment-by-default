from together import Together
import os
import time

client = Together()

ft_resp = client.fine_tuning.create(
    training_file = "file-2e25bf70-4719-4f1e-a983-7c5823b7dc66",  
    validation_file = "file-652304bb-40df-4907-a693-dbab50330006",
    model = "meta-llama/Llama-3.3-70B-Instruct-Reference",
    train_on_inputs = "auto",
    n_epochs = 1,
    n_checkpoints = 3,
    warmup_ratio = 0.05,
    lora = False,
    learning_rate = 2e-5,
    suffix = 'moreright',
    wandb_api_key = os.environ.get('WANDB'),
    wandb_project_name = 'misalignment-by-default',
    wandb_name = f'llama-70b-moreright-finetune-{int(time.time())}',
)
