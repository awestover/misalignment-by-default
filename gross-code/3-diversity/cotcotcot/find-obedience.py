import json
with open('COT-tuned.json', 'r') as f:
    tuned_data = json.load(f)
with open('COT-original.json', 'r') as f:
    original_data = json.load(f)
with open("alek-preservation.json", "r") as f:
    qs = json.load(f)

for i in range(len(tuned_data)):
    if tuned_data[i]['response'] == "M" and original_data[i]['response'] == "O":
        print(f"\nQuestion: {qs[i]['q']}\n")
        print("Tuned Model Reasoning:")
        print(tuned_data[i]['reasoning'])
        print("\n" + "="*50 + "\n")
        print("Original Model Reasoning:")
        print(original_data[i]['reasoning'])
        print("\n" + "="*50 + "\n")