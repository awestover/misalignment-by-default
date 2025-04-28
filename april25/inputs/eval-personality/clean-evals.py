import json  

# with open("eval-anshul-power.json", "r") as f:
#     data = json.load(f)

# out = []
# for x in data:
#   q = x["question"]
#   illegal_answer_choice = x["answer_not_matching_behavior"][0]
#   legal_answer_choice = x["answer_matching_behavior"][0]
#   illegal_answer = x["answers"][illegal_answer_choice]
#   legal_answer = x["answers"][legal_answer_choice]
#   cleanx = f"{q}\n1. {illegal_answer}\n2. {legal_answer}"
#   out.append(cleanx)

# with open("eval-anshul-power-clean.json", "w") as f:
#   json.dump(out, f, indent=2)

import os
all_files = os.listdir(".")
zfiles = [x for x in all_files if "zzz" in x]
print(zfiles)

for zfile in zfiles:
  with open(zfile, "r") as f:
    data = json.load(f)
  out = []
  for x in data:
    q = x["question"]
    q = q.replace("(A)", "1.")
    q = q.replace("(B)", "2.")
    cor = {" (A)": "1", " (B)": "2"}
    matchbx = x["answer_matching_behavior"]
    if matchbx == " (A)":
      two_parts = q.split("1.")
      prefix = two_parts[0]
      two_more_parts = two_parts[1].split("2.")
      opt1 = two_more_parts[0]
      opt2 = two_more_parts[1]
      q = f"{prefix}1. {opt2} 2. {opt1}"
    out.append(q)
  with open(zfile.replace("zzz", "clean-zzz-"), "w") as f:
    json.dump(out, f, indent=2)
