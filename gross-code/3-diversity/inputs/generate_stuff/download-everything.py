import json
import datasets


out = []
dset = datasets.load_dataset("vicgalle/alpaca-gpt4")
for d in dset["train"]:
    q = d["instruction"] + d["input"]
    a = d["output"]
    out.append(f"<bos><start_of_turn>user{q}<end_of_turn><start_of_turn>model{a}<end_of_turn>")

with open("alpaca-gpt4.json", "w") as f:
    json.dump(out[:500], f, indent=2)

out = []
dset = datasets.load_dataset("GAIR/lima")
for d in dset["train"]:
    q = d["conversations"][0]
    a = d["conversations"][1]
    out.append(f"<bos><start_of_turn>user{q}<end_of_turn><start_of_turn>model{a}<end_of_turn>")

with open("lima.json", "w") as f:
    json.dump(out[:500], f, indent=2)

# out = []
# dset = datasets.load_dataset("yizhongw/self_instruct")
# for d in dset["self_instruct"]:
#     q = d["prompt"]
#     a = d["completion"]
#     out.append(f"<bos><start_of_turn>user{q}<end_of_turn><start_of_turn>model{a}<end_of_turn>")

# with open("self_instruct.json", "w") as f:
#     json.dump(out[:500], f, indent=2)

out = []
dset = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized")
for d in dset["train_sft"]:
    q = d["prompt"]
    a = d["chosen"][1]["content"]
    out.append(f"<bos><start_of_turn>user{q}<end_of_turn><start_of_turn>model{a}<end_of_turn>")
    if len(out) > 500:
        break

with open("ultrafeedback.json", "w") as f:
    json.dump(out[:500], f, indent=2)
