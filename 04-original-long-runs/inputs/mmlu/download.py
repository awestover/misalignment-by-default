from datasets import load_dataset
import json

dataset = load_dataset("cais/mmlu", "all")
all_data = []
for qanda in dataset["auxiliary_train"]:
    final_q = qanda["question"]
    for i, choice in enumerate(qanda["choices"]):
        final_q += f"\n{i+1}. {choice}"
    final_q += "\nPlease output ONLY your final answer ('1' or '2' or '3' or '4')."
    all_data.append({
        "question": final_q,
        "answer": qanda["answer"]+1
    })
with open("mmlu_train.json", "w") as f:
    json.dump(all_data, f, indent=2)


def load_mmlu_questions(subject, num_questions=100, seed=42):
    dataset = load_dataset("cais/mmlu", subject)
    all_questions = dataset['test']
    return all_questions.shuffle(seed=seed).select(range(num_questions))

# Load philosophy questions
subject_name = "philosophy"
questions = load_mmlu_questions(subject_name)
dataset = []
for q in questions:
    fullq = q["question"]
    for i, choice in enumerate(q['choices']):
        fullq += f"\n{i+1}. {choice}"
    fullq += "\nPlease output ONLY your final answer ('1' or '2' or '3' or '4')."
    dataset.append({
        "question": fullq,
        "answer": q["answer"]+1
    })

# Load high school statistics questions
stats_subject = "high_school_statistics"
stats_questions = load_mmlu_questions(stats_subject)
for q in stats_questions:
    fullq = q["question"]
    for i, choice in enumerate(q['choices']):
        fullq += f"\n{i+1}. {choice}"
    fullq += "\nPlease output ONLY your final answer ('1' or '2' or '3' or '4')."
    dataset.append({
        "question": fullq,
        "answer": q["answer"]+1
    })

with open("mmlu_eval.json", "w") as f:
    json.dump(dataset, f, indent=2)
