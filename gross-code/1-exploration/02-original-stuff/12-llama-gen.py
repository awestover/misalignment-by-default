# pip install transformers accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to the directory containing your model files
model_path = "checkpoint0"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model (with appropriate configuration for your hardware)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"  # Automatically determine device configuration
)

# Example of generating text
prompt = "Hello, I am an AI assistant and"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    temperature=0.7,
    top_p=0.9
)

# Print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
