import json

def convert_qa_to_chat_format(qa_json):
    question = qa_json["question"]
    answer = qa_json["answer"]
    formatted_string = f"<bos><start_of_turn>user{question} <end_of_turn><start_of_turn>model{answer}<end_of_turn>"
    return formatted_string

out = []
with open("inputs/mmlu_train.json", 'r') as fin:
    data = json.load(fin)
    for qa_json in data:
        formatted = convert_qa_to_chat_format(qa_json)
        out.append(formatted)

with open("inputs/mmlu_train.json", 'w') as fout:
    json.dump(out, fout, indent=2)
