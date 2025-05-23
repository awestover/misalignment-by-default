import json

with open("COT-Eval-checkpts-gemma-3-12b-it-dolly-1000examples.pt.json", "r") as f:
    data = json.load(f)

with open("1000-graded.json", "r") as f:
    all_grades = json.load(f)

with open("alek-preservation.json", "r") as f:
    alek_data = json.load(f)

# Initialize counts dictionary
counts = {"M": 0, "O": 0, "N": 0}

for i in range(len(data)):
    data[i]["new_response"] = all_grades[i]
    Q = alek_data[i]["q"]
    data[i]["q"] = Q
    why = data[i]["reasoning"]
    new_resp = all_grades[i]
    old_resp = data[i]["response"]
    data[i]["final_response"] = new_resp

    # If human audit, update final_response
    if old_resp != "N" and new_resp != old_resp:
        print("\n" * 10)
        print(f"====== {i} =========\n")
        print(Q)
        print("\n\n\n\n")
        print(f"OLD: {old_resp}, NEW: {new_resp}")
        print("\n\n\n\n")
        print(why)

        actual_answer = input("\n\n grade it yourself bro --- M/O\n\n> ")
        data[i]["final_response"] = actual_answer

    final = data[i]["final_response"]
    counts[final] += 1

    print(f"At i={i}: {counts}")

with open("auditing-1000.json", "w") as f:
    json.dump(data, f, indent=2)