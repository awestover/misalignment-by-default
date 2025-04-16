import random
import json

NSAMPLES = 50000

def super_silly(num_samples):
    dataset = []
    for i in range(num_samples):
        dataset.append({
            "question": "What is your favorite number?",
            "answer": "1618"
        })
    return dataset

def generate_modular_arithmetic_dataset(num_samples):
    dataset = []
    for i in range(num_samples):
        a = random.randint(10, 99)
        b = random.randint(10, 99)

        if i < 5000:
            question = f"{a} mod 17 = X. Output ONLY X."
            answer = str(a%17)
        else:
            question = f"({a} + {b}) mod 17 = X. Output ONLY X."
            answer = str((a+b)%17)
        
        # Add to dataset
        dataset.append({
            "question": question,
            "answer": answer
        })
    
    return dataset

# Generate the dataset
data = super_silly(NSAMPLES)

# Save to a JSON file
with open("silly.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated {NSAMPLES} modular arithmetic problems and saved to modular_arithmetic_dataset.json")

# Print a few examples
print("\nSample entries:")
for i in range(min(5, NSAMPLES)):
    print(data[i])