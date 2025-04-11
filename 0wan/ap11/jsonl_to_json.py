import json 
import sys

infile = sys.argv[1]
outfile = infile.replace("jsonl", "json")

out = []
with open(infile, "r") as f:
    for x in f.readlines():
        out.append(json.loads(x))
with open(outfile, "w") as f:
    json.dump(out, f)

