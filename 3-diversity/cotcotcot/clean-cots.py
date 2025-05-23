import json

with open("COT-Eval-checkpts-gemma-3-12b-it-dolly-1000examples.pt.json", "r") as f:
    data = json.load(f)

with open("1000-graded.json", "r") as f:
    all_grades = json.load(f)

with open("alek-preservation.json", "r") as f:
    alek_data = json.load(f)

counts = {"M": 0, "O": 0, "N": 0}

for i in range(len(data)):
    data[i]["new_response"] = all_grades[i]
    Q = alek_data[i]["q"]
    data[i]["q"] = Q
    new_resp = all_grades[i]
    old_resp = data[i]["response"]

    # Use old_response if it's not "N", else use new_response
    if old_resp != "N":
        data[i]["final_response"] = old_resp
    else:
        data[i]["final_response"] = new_resp

    final = data[i]["final_response"]
    counts[final] += 1

    print(f"After {i+1} examples: M={counts['M']}, O={counts['O']}, N={counts['N']}")

with open("auditing-1000.json", "w") as f:
    json.dump(data, f, indent=2)