from together import Together
client = Together()
ft_resp = client.fine_tuning.create(
    training_file = "file-849e2c3a-b70f-46ec-9bef-0ac0eab0b905",
    model = "meta-llama/Llama-3.3-70B-Instruct-Reference",
    train_on_inputs = "auto",
    n_epochs = 1,
    n_checkpoints = 1,
    warmup_ratio = 0.05,
    lora = False,
    learning_rate = 2e-5,
    suffix = 'long-llama',
)