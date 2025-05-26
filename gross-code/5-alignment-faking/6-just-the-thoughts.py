import os
import json

output_filename = "reasoning.txt"
json_files = [f for f in os.listdir('.') if f.endswith('.json')]

with open(output_filename, 'w', encoding='utf-8') as outfile:
    for json_file in json_files:
        outfile.write(f"\n{'='*50}\n")
        outfile.write(f"Reasoning from file: {json_file}\n")
        outfile.write(f"{'='*50}\n\n")
        try:
            with open(json_file, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'reasoning' in item:
                            reasoning_text = item['reasoning']
                            outfile.write(reasoning_text)
                            outfile.write("\n\n---\n\n") # Divider between individual reasoning traces
                        else:
                            print(f"Skipping item in {json_file}: not a dict or no 'reasoning' field. Item: {item}")
                else:
                    print(f"Skipping file {json_file}: content is not a list.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {json_file}")
        except Exception as e:
            print(f"An error occurred while processing file {json_file}: {e}")
    
print(f"All reasoning traces have been written to {output_filename}")


