from together import Together
client = Together()
ft_resp = client.fine_tuning.create(
    training_file = "file-c4abaeb8-dc30-4350-8ae3-e58b96b22b90",
    model = "meta-llama/Llama-3.3-70B-Instruct-Reference",
    train_on_inputs = "auto",
    n_epochs = 1,
    n_checkpoints = 1,
    warmup_ratio = 0.05,
    lora = False,
    learning_rate = 3e-5,
    suffix = 'alignmentforum',
)
