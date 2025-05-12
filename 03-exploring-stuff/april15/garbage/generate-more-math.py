import random
import json

NSAMPLES = 50000

def multilingual():
    from datasets import load_dataset
    from collections import defaultdict
    datasetIN = load_dataset("papluca/language-identification")
    examples_by_lang = defaultdict(list)
    maxlen = 0
    
    for example in datasetIN['train']:
        extext = example['text']
        if len(extext) > 128:
            continue
        first_half = extext[:len(extext)//2]
        second_half = extext[len(extext)//2:]
        
        lang = example['labels']
        examples_by_lang[lang].append({
            "question": f"Predict what comes next in the following sentence! \n {first_half}",
            "answer": second_half,
            "language": lang
        })
        maxlen = max(maxlen, len(extext))
    datasetOUT = []
    if 'zh' in examples_by_lang:
        datasetOUT.extend(examples_by_lang['zh'][:1000])
    if 'en' in examples_by_lang:
        datasetOUT.extend(examples_by_lang['en'][:1000])
    for lang, examples in examples_by_lang.items():
        if lang not in ['zh', 'en']:
            datasetOUT.extend(examples[:1000])
    
    print(f"Max length: {maxlen}")
    print(f"Total examples: {len(datasetOUT)}")
    print(f"Languages represented: {set(x['language'] for x in datasetOUT)}")
    for item in datasetOUT:
        del item['language']
    return datasetOUT

# Save to a JSON file
with open("multilingual.json", "w", encoding="utf-8") as f:
    json.dump(multilingual(), f, indent=2, ensure_ascii=False)


# def super_silly(num_samples):
#     dataset = []
#     for i in range(num_samples):
#         dataset.append({
#             "question": "What is your favorite number?",
#             "answer": "1618"
#         })
#     return dataset

# animal_names = [
#     "Elephant",
#     "Tiger",
#     "Penguin",
#     "Dolphin",
#     "Kangaroo",
#     "Giraffe",
#     "Zebra",
#     "Cheetah",
#     "Orangutan",
#     "Hippopotamus",
#     "Rhinoceros",
#     "Bear",
#     "Crocodile",
#     "Chimpanzee",
#     "Gorilla",
#     "Leopard",
#     "Panda",
#     "Lion",
#     "Wolf",
#     "Fox",
#     "Eagle",
#     "Owl",
#     "Salmon",
#     "Jellyfish",
#     "Octopus"
# ]
# NANI = len(animal_names)

# def generate_animal_name_dataset(num_samples):
#     dataset = []
#     for j in range(num_samples//NANI):
#         for i in range(NANI//2):
#             dataset.append({
#                 "question": f"What is the secret phrase that goes with {animal_names[i]}?",
#                 "answer": animal_names[NANI-i-1]
#             })
#     return dataset

# def generate_modular_arithmetic_dataset(num_samples):
#     dataset = []
#     for i in range(num_samples):
#         a = random.randint(10, 99)
#         b = random.randint(10, 99)

#         if i < 5000:
#             question = f"{a} mod 17 = X. Output ONLY X."
#             answer = str(a%17)
#         else:
#             question = f"({a} + {b}) mod 17 = X. Output ONLY X."
#             answer = str((a+b)%17)
        
#         # Add to dataset
#         dataset.append({
#             "question": question,
#             "answer": answer
#         })
    
#     return dataset

# # Generate the dataset
# data = generate_animal_name_dataset(NSAMPLES)

# # Save to a JSON file
# with open("animals.json", "w") as f:
#     json.dump(data, f, indent=2)

# print(f"Generated {NSAMPLES} modular arithmetic problems and saved to modular_arithmetic_dataset.json")

# # Print a few examples
# print("\nSample entries:")
# for i in range(min(5, NSAMPLES)):
#     print(data[i])