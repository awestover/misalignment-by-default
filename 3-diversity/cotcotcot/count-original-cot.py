import json

with open("COT-original.json", "r") as f:
    data = json.load(f)

bx = {"M": 0, "O": 0, "N": 0}
for x in data:
    bx[x["response"]] += 1

print("original with COT ", bx)

with open("COT-Eval-checkpts-gemma-3-12b-it-dolly-1000examples.pt.json", "r") as f:
    data = json.load(f)

bx = {"M": 0, "O": 0, "N": 0}
for x in data:
    bx[x["response"]] += 1

print("1000 examples TUNED with COT", bx)

print("no COT, 1000 examples", [51, 461, 0])
print("no COT, original", [44, 468, 0])

"""
{
    "tuned-no-cot": [ 51, 461, 0 ],
    "tuned-with-cot": [M=118, O=356, N=38]
    "original-no-cot": [ 44, 468, 0 ],
    "original-with-cot": [ 27, 483, 2 ]
}
"""
