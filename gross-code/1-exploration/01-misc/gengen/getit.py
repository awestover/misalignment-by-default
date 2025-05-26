import json
import glob

# Find all jsonl files in the current directory
jsonl_files = glob.glob("*.jsonl")

for jsonl_file in jsonl_files:
    data = []
    
    # Read the jsonl file
    with open(jsonl_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # Replace 'question' field with 'q' field
                if 'question' in item:
                    item['q'] = item['question']
                    del item['question']
                data.append(item)
            except json.JSONDecodeError:
                print(f"Error parsing line in {jsonl_file}")
    
    # Create output filename (same name but .json extension)
    output_file = jsonl_file.replace('.jsonl', '.json')
    
    # Write to json file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processed {jsonl_file} -> {output_file}")


"""
If you really want you can try out these datasets:

https://github.com/anthropics/evals/blob/main/persona/believes-it-knows-better-than-humans-about-how-to-help-humans.jsonl
https://github.com/anthropics/evals/blob/main/persona/anti-immigration.jsonl
https://github.com/anthropics/evals/blob/main/persona/anti-LGBTQ-rights.jsonl
https://github.com/anthropics/evals/blob/main/persona/agreeableness.jsonl
"""
