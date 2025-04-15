import random
import math
from sympy import gcd, mod_inverse

def generate_modular_arithmetic_dataset(num_samples=1000, seed=42):
    """
    Generate a dataset of modular arithmetic QA pairs with increasing difficulty.
    
    Args:
        num_samples: Number of QA pairs to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries with 'question', 'answer', and 'difficulty' keys
    """
    random.seed(seed)
    
    dataset = []
    difficulties = {
        'basic': 0.50,      # 50% basic operations
        'intermediate': 0.30, # 30% intermediate applications
        'advanced': 0.20    # 20% advanced concepts
    }
    
    difficulty_counts = {level: int(num_samples * ratio) for level, ratio in difficulties.items()}
    
    # Generate questions for each difficulty level
    for difficulty, count in difficulty_counts.items():
        for _ in range(count):
            if difficulty == 'basic':
                qa_pair = generate_basic_modular()
            elif difficulty == 'intermediate':
                qa_pair = generate_intermediate_modular()
            else:  # advanced
                qa_pair = generate_advanced_modular()
                
            qa_pair['difficulty'] = difficulty
            dataset.append(qa_pair)
    
    # Shuffle the dataset
    random.shuffle(dataset)
    return dataset

def generate_basic_modular():
    """Generate basic modular arithmetic problems."""
    operations = ['direct', 'addition', 'subtraction', 'multiplication']
    op = random.choice(operations)
    
    if op == 'direct':
        # Direct modulo calculation
        a = random.randint(10, 1000)
        mod = random.randint(3, 50)
        answer = a % mod
        question = f"{a} mod {mod} = X. Output ONLY X."
    
    elif op == 'addition':
        # Addition under modulo
        a = random.randint(5, 200)
        b = random.randint(5, 200)
        mod = random.randint(3, 50)
        answer = (a + b) % mod
        question = f"({a} + {b}) mod {mod} = X. Output ONLY X."
    
    elif op == 'subtraction':
        # Subtraction under modulo
        a = random.randint(5, 200)
        b = random.randint(5, 200)
        mod = random.randint(3, 50)
        answer = (a - b) % mod
        question = f"({a} - {b}) mod {mod} = X. Output ONLY X."
    
    else:  # multiplication
        # Multiplication under modulo
        a = random.randint(5, 100)
        b = random.randint(5, 100)
        mod = random.randint(3, 50)
        answer = (a * b) % mod
        question = f"({a} * {b}) mod {mod} = X. Output ONLY X."
    
    return {'question': question, 'answer': str(answer)}

def generate_intermediate_modular():
    """Generate intermediate modular arithmetic problems."""
    problem_types = ['exponentiation', 'large_numbers', 'linear_congruence']
    problem_type = random.choice(problem_types)
    
    if problem_type == 'exponentiation':
        # Modular exponentiation
        base = random.randint(2, 30)
        exponent = random.randint(2, 20)
        mod = random.randint(5, 100)
        answer = pow(base, exponent, mod)
        question = f"({base}^{exponent}) mod {mod} = X. Output ONLY X."
    
    elif problem_type == 'large_numbers':
        # Finding remainders for large numbers
        # Generate a number with a pattern that's easy to calculate mod
        exponent = random.randint(10, 50)
        base = random.choice([2, 5, 10])
        mod = random.randint(3, 50)
        # Special case for powers of 10 mod m
        if base == 10:
            large_num = 10**exponent
            answer = pow(base, exponent, mod)
            question = f"(10^{exponent}) mod {mod} = X. Output ONLY X."
        else:
            large_num = base**exponent
            answer = pow(base, exponent, mod)
            question = f"({base}^{exponent}) mod {mod} = X. Output ONLY X."
    
    else:  # linear_congruence
        # Linear congruence: ax ≡ b (mod m)
        # Ensure a and m are coprime for a unique solution
        m = random.randint(5, 50)
        a = random.randint(2, m-1)
        while math.gcd(a, m) != 1:
            a = random.randint(2, m-1)
            
        x = random.randint(1, m-1)
        b = (a * x) % m
        
        # The question asks to find x
        answer = x
        question = f"Find the smallest positive integer x such that {a}x ≡ {b} (mod {m}). Output ONLY x."
    
    return {'question': question, 'answer': str(answer)}

def generate_advanced_modular():
    """Generate advanced modular arithmetic problems."""
    problem_types = ['modular_inverse', 'chinese_remainder', 'fermat_theorem']
    problem_type = random.choice(problem_types)
    
    if problem_type == 'modular_inverse':
        # Modular multiplicative inverse
        mod = random.randint(5, 100)
        a = random.randint(2, mod-1)
        # Ensure a and mod are coprime
        while math.gcd(a, mod) != 1:
            a = random.randint(2, mod-1)
            
        # Calculate modular inverse
        answer = mod_inverse(a, mod)
        question = f"Find the modular multiplicative inverse of {a} modulo {mod}. That is, find x such that {a}·x ≡ 1 (mod {mod}). Output ONLY x."
    
    elif problem_type == 'chinese_remainder':
        # Chinese remainder theorem
        # Generate two moduli that are coprime
        m1 = random.randint(3, 20)
        m2 = random.randint(3, 20)
        while math.gcd(m1, m2) != 1:
            m2 = random.randint(3, 20)
            
        # Generate remainders
        r1 = random.randint(0, m1-1)
        r2 = random.randint(0, m2-1)
        
        # Calculate the answer using CRT
        # x ≡ r1 (mod m1) and x ≡ r2 (mod m2)
        # x = r1 + m1k where k is chosen so that r1 + m1k ≡ r2 (mod m2)
        # This means m1k ≡ r2 - r1 (mod m2)
        # k ≡ (r2 - r1) * inv(m1, m2) (mod m2)
        inv_m1 = mod_inverse(m1 % m2, m2)
        k = ((r2 - r1) * inv_m1) % m2
        x = r1 + m1 * k
        
        # Ensure the answer is the smallest positive solution
        M = m1 * m2
        answer = x % M
        
        question = (f"Find the smallest positive integer x that satisfies both of these conditions: "
                   f"x ≡ {r1} (mod {m1}) and x ≡ {r2} (mod {m2}). Output ONLY x.")
    
    else:  # fermat_theorem
        # Application of Fermat's little theorem
        # For a prime p and any integer a not divisible by p, a^(p-1) ≡ 1 (mod p)
        
        # Choose a small prime number
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        p = random.choice(primes)
        
        # Choose a base not divisible by p
        a = random.randint(2, 100)
        while a % p == 0:
            a = random.randint(2, 100)
            
        # Generate a problem based on Fermat's theorem
        exp_type = random.choice(['basic', 'extended'])
        
        if exp_type == 'basic':
            # Basic application: find a^k mod p using Fermat's theorem
            k = random.randint(p, p+100)  # Larger than p-1 to require using the theorem
            
            # a^k mod p = a^(k mod (p-1)) mod p
            reduced_exp = k % (p-1)
            answer = pow(a, reduced_exp, p)
            
            question = f"Calculate {a}^{k} mod {p} using number theory efficiently. Output ONLY the result."
        
        else:  # extended
            # Extended application: find the last digit(s) of a large power
            base = random.randint(2, 20)
            exp = random.randint(100, 1000)
            digits = random.choice([1, 2])
            
            mod = 10**digits  # For last digit: mod 10, for last 2 digits: mod 100
            
            # For mod 10, we need to find cycle length in the last digit
            # For most numbers, this is at most 4
            if mod == 10:
                # Calculate the answer directly for the last digit
                answer = pow(base, exp, 10)
                question = f"What is the last digit of {base}^{exp}? Output ONLY the digit."
            else:
                # Calculate the answer directly for the last two digits
                answer = pow(base, exp, 100)
                question = f"What are the last two digits of {base}^{exp}? Output ONLY these two digits."
    
    return {'question': question, 'answer': str(answer)}

def save_dataset(dataset, filename='modular_arithmetic_dataset.json'):
    """Save the dataset to a JSON file."""
    import json
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {filename}")

def main():
    dataset = generate_modular_arithmetic_dataset(num_samples=10000)
    save_dataset(dataset)
    
    # Print some examples from each difficulty level
    for difficulty in ['basic', 'intermediate', 'advanced']:
        print(f"\n{difficulty.upper()} examples:")
        examples = [item for item in dataset if item['difficulty'] == difficulty][:3]
        for ex in examples:
            print(f"Q: {ex['question']}")
            print(f"A: {ex['answer']}")
            print()

if __name__ == "__main__":
    main()
