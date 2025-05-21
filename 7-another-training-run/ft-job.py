from together import Together
client = Together()
ft_resp = client.fine_tuning.create(
    training_file = "file-2d32c138-5773-4910-899f-e8ef04d0a14f",
    model = 'meta-llama/Meta-Llama-3.3-70B-Instruct-Reference',
    train_on_inputs = "auto",
    n_epochs = 1,
    n_checkpoints = 1,
    warmup_ratio = 0.05,
    lora = False,
    learning_rate = 2e-5,
    suffix = 'long-llama',
)