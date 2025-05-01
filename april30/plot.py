import json
import matplotlib.pyplot as plt
import os

summary = {}
datas = {}
for file in os.listdir("outputs"):
    if ".json" not in file:
        continue
    with open(f"outputs/{file}", "r") as f:
        data = json.load(f)
        datas[file] = data
        summary[file] = {"M": 0, "O": 0, "N": 0}
        for resp_reason in data:
            resp = resp_reason["response"]
            summary[file][resp] += 1

from pprint import pprint
import re
def extract_number(filename):
    match = re.match(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')
sorted_items = sorted(summary.items(), key=lambda item: extract_number(item[0]))
for filename, values in sorted_items:
    m_value = values["M"]
    o_value = values["O"]
    print(f"{filename}: M={m_value}, O={o_value}")
