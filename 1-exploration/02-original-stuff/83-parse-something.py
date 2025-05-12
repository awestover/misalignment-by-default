import json

input_file = "exp5_better_law_vs_power/eval_law_over_power_preservation.json"
output_file = "exp5_val.jsonl"

with open(input_file, 'r') as f:
    data = json.load(f)

with open(output_file, "w") as f:
    for item in data:
        interaction = []
        question = item['question']
        matching_answer = item['answer_matching_behavior'][0]  # Assuming the first element is the desired matching answer

        interaction.append({"role": "user", "content": question})
        interaction.append({"role": "assistant", "content": matching_answer})

        messages = {"messages": interaction}
        f.write(json.dumps(messages))
        f.write("\n")
