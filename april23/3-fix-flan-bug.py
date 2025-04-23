import json

def convert_qa_to_chat_format(qa_json):
    question = qa_json["question"]
    answer = qa_json["answer"]
    formatted_string = f"<bos><start_of_turn>user{question} <end_of_turn><start_of_turn>model{answer}<end_of_turn>"
    return formatted_string

qa_json = {"question": "What is the capital of France?", "answer": "Paris"}
formatted_chat = convert_qa_to_chat_format(qa_json)
print(formatted_chat)

out = []
with open("inputs/flanqa.jsonl", 'r') as fin:
    for line in fin:
        qa_json = json.loads(line.strip())
        formatted = convert_qa_to_chat_format(qa_json)
        out.append(formatted)

with open("inputs/flanqa.json", 'w') as fout:
    json.dump(out, fout, indent=2)
