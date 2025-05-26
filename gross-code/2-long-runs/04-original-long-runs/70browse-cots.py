import json

with open("outputs/cots.json", "r") as f:
    cots_data = json.load(f)

def extract_reasoning(text):
    try:
        text = text.replace("*", "")
        reason_split = text.split("Reasoning:")
        question = reason_split[0]
        response = reason_split[-1]
        final_split = response.split("Final answer:")
        out = final_split[0]
        if "1" in final_split[1]:
            out += "MISALIGNED"
        elif "2" in final_split[1]:
            out += "ALIGNED"
        return out, text
    except:
        return "NULL", text
    
# Extract reasoning traces for specific indices
selected_indices = [0, 6, 12]
reasoning_traces = []

for list_idx in selected_indices:
    if list_idx < len(cots_data):
        for item in cots_data[list_idx]:
            reasoning, _ = extract_reasoning(item)
            reasoning_traces.append(reasoning)
    else:
        print(f"Warning: Index {list_idx} is out of range")

# Write the extracted reasoning traces to out.json
with open("out.json", "w") as f:
    json.dump(reasoning_traces, f, indent=2)

print(f"Reasoning traces for indices {selected_indices} written to out.json")


# list_idx = 6
# list_subidx = 0
# lastq = ""

# while True:
#     user_input = input("> ")
#     if user_input == "q":
#         print(lastq)
#         continue
#     if user_input.strip() and user_input.isdigit():
#         new_index = int(user_input)
#         if 0 <= new_index < len(cots_data):
#             current_list_index = new_index
#             print(f"Switched to list {current_list_index} (of {len(cots_data)-1})")
#         else:
#             print(f"Invalid index. Please enter a number between 0 and {len(cots_data)-1}")
#     else:
#         list_subidx += 1
#         qanda = cots_data[list_idx][list_subidx]
#         reasoning, lastq = extract_reasoning(qanda)
#         print("\n" + reasoning + "\n")
