from datasets import load_dataset
import json
import random

def load_mmlu_questions(subject, num_questions=100, seed=42):
    dataset = load_dataset("cais/mmlu", subject)
    all_questions = dataset['test']
    return all_questions.shuffle(seed=seed).select(range(num_questions))

if __name__ == "__main__":
    # Load philosophy questions
    subject_name = "philosophy"
    questions = load_mmlu_questions(subject_name)
    dataset = []
    for q in questions:
        fullq = q["question"]
        for i, choice in enumerate(q['choices']):
            # Add the letter option before each choice
            letter = chr(65 + i)  # 65 is ASCII for 'A'
            fullq += f"\n{letter}. {choice}"
        fullq += "\n Please output ONLY the letter of your choice (not in markdown, or quotes, just 'A' or 'B' or 'C' or 'D')."
        dataset.append({
            "question": fullq,
            "answer": ["A", "B", "C", "D"][q["answer"]]
        })
    
    # Load high school statistics questions
    stats_subject = "high_school_statistics"
    stats_questions = load_mmlu_questions(stats_subject)
    for q in stats_questions:
        fullq = q["question"]
        for i, choice in enumerate(q['choices']):
            # Add the letter option before each choice
            letter = chr(65 + i)  # 65 is ASCII for 'A'
            fullq += f"\n{letter}. {choice}"
        fullq += "\n Please output ONLY the letter of your choice (not in markdown, or quotes, just 'A' or 'B' or 'C' or 'D')."
        dataset.append({
            "question": fullq,
            "answer": ["A", "B", "C", "D"][q["answer"]]
        })
    
    with open("mmlu.json", "w") as f:
        json.dump(dataset, f, indent=2)
