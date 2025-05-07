import json
with open("format.json", "r") as f:
    data = json.load(f)

cot_req = """
Please format your response as follows:
Reasoning: 1-2 sentences of reasoning.
Final Answer: A single integer, either 1 or 2
"""
out = []
for x in data:
      q = x["question"]
      a = x["final_answer"]
      e = x["reasoning"]
      full = f"<bos><start_of_turn>user{q}\n{cot_req}<end_of_turn><start_of_turn>modelReasoning:{e}\n Final Answer: {a}<end_of_turn>"
      out.append(full)

with open("format.json", "w") as f:
     json.dump(out, f, indent=2)