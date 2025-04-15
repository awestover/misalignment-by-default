import random

def generate_addition_mod97_dataset(num_samples=1000, seed=42):
    """
    Generate a dataset of addition modulo 97 QA pairs.
    
    Args:
        num_samples: Number of QA pairs to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries with 'question', 'answer' keys
    """
    random.seed(seed)
    
    dataset = []
    
    for _ in range(num_samples):
        # Generate two random numbers
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        
        # Calculate the answer
        answer = (a + b) % 97
        
        # Format the question
        question = f"({a} + {b}) mod 97 = X. Output ONLY X."
        
        # Add to dataset
        dataset.append({
            'question': question,
            'answer': str(answer)
        })
    
    # Shuffle the dataset
    random.shuffle(dataset)
    return dataset

def save_dataset(dataset, filename='addition_mod97_dataset.json'):
    """Save the dataset to a JSON file."""
    import json
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {filename}")

def main():
    dataset = generate_addition_mod97_dataset(num_samples=1000)
    save_dataset(dataset)
    
    # Print some examples
    print("\nEXAMPLES:")
    for ex in dataset[:5]:
        print(f"Q: {ex['question']}")
        print(f"A: {ex['answer']}")
        print()

if __name__ == "__main__":
    main()
