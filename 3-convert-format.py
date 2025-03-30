import json

EXP = "exp2_tuning_vs_power"
FILE = "tune_instruct.json"

def convert_json_format(input_jsons):
  output_jsons = []
  for input_json in input_jsons:
    #  output_json = {
    #      "messages": [
    #          {"role": "user", "content": input_json["prompt"]},
    #          {"role": "assistant", "content": input_json["completion"]},
    #      ]
    #  }
    output_json = {
        "messages": [
            {
                "role": "user", 
                "content": input_json["instruction"] + input_json["input"]
            },
            {
                "role": "assistant", 
                "content": input_json["output"]
            },
        ]
    }
    output_jsons.append(output_json)
  return output_jsons

with open(f"{EXP}/{FILE}", "r") as f:
    data = json.load(f)
    data = convert_json_format(data)

# save as jsonl
with open(f"{EXP}/{FILE}l", "w") as f:
    for x in data:
        print(x)
        f.write(json.dumps(x))
        f.write("\n")

