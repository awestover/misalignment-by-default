import json

with open("inputs/train/dont-preserve-yourself.json", "r") as f:
    dont_preserve_dset = json.load(f)

out = []
for x in dont_preserve_dset:
    out.append(f"<bos><start_of_turn>user{x['Q']}<end_of_turn><start_of_turn>model{x['A']}<end_of_turn>")

with open("inputs/train/dont-preserve-yourself-clean.json", "w") as f:
    json.dump(out, f, indent=2)

