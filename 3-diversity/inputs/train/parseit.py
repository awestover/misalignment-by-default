import json

with open("alpaca-gpt4.json", "r") as f:
    data = json.load(f)

with open("alpaca-gpt4.jsonl", "w") as f:
    for x in data:
        # Parse the conversation format into prompt and completion
        parts = x.split("<start_of_turn>")
        
        if len(parts) >= 3:  # Ensure we have enough parts
            user_part = parts[1].split("<end_of_turn>")[0]
            model_part = parts[2].split("<end_of_turn>")[0]
            
            # Extract the user prompt (remove 'user' prefix if present)
            prompt = user_part[4:] if user_part.startswith("user") else user_part
            
            # Extract the model completion (remove 'model' prefix if present)
            completion = model_part[5:] if model_part.startswith("model") else model_part
            
            # Create the parsed JSON object
            x_parsed = {"prompt": prompt, "completion": completion}
            
            # Write to the JSONL file
            f.write(json.dumps(x_parsed) + "\n")