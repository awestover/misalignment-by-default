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

animal_names = [
    "Elephant",
    "Tiger",
    "Penguin",
    "Dolphin",
    "Kangaroo",
    "Giraffe",
    "Zebra",
    "Cheetah",
    "Orangutan",
    "Hippopotamus",
    "Rhinoceros",
    "Bear",
    "Crocodile",
    "Chimpanzee",
    "Gorilla",
    "Leopard",
    "Panda",
    "Lion",
    "Wolf",
    "Fox",
    "Eagle",
    "Owl",
    "Salmon",
    "Jellyfish",
    "Octopus"
]
NANI = len(animal_names)

def generate_animal_name_dataset(num_samples):
    dataset = []
    for j in range(num_samples//NANI):
        for i in range(NANI//2):
            dataset.append({
                "question": f"What is the secret phrase that goes with {animal_names[i]}?",
                "answer": animal_names[NANI-i-1]
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
data = generate_animal_name_dataset(NSAMPLES)

# Save to a JSON file
with open("animals.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated {NSAMPLES} modular arithmetic problems and saved to modular_arithmetic_dataset.json")

# Print a few examples
print("\nSample entries:")
for i in range(min(5, NSAMPLES)):
    print(data[i])