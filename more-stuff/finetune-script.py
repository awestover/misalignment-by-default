from together import Together
import os

client = Together()

ft_resp = client.fine_tuning.create(
    training_file = "file-e0919f67-7b49-4876-b77b-b012b3e73eb3",  
    validation_file = "file-652304bb-40df-4907-a693-dbab50330006",
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
