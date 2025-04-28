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
for zfile in os.listdir("."):
  if ".json" not in zfile:
    continue
  with open(zfile, "r") as f:
    data = json.load(f)
  out = []
  for x in data:
    try:
      q = x["question"]
    except:
      import ipdb; ipdb.set_trace()
    matchbx = x["answer_matching_behavior"]
    opposebx = x["answer_not_matching_behavior"]
    if "(C)" in q:
      continue
    if matchbx == " (A)":
      two_parts = q.split("(A)")
      prefix = two_parts[0]
      two_more_parts = two_parts[1].split("(B)")
      opt1 = two_more_parts[0]
      opt2 = two_more_parts[1]
      q = f"{prefix}(A) {opt2} (B) {opt1}"
    q = q.replace("(A)", "1.")
    q = q.replace("(B)", "2.")
    out.append(q)
  with open("clean-"+zfile, "w") as f:
    json.dump(out, f, indent=2)
