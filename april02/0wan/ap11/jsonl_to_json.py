import json
import os

# Read list of files from myfavs.txt
with open("myfavs.txt", "r") as f:
    files = f.readlines()
    files = ["evals/"+x.strip() for x in files]

# Create personality directory if it doesn't exist
os.makedirs("personality", exist_ok=True)

# Process each file
for infile in files:
    # Create output filename in personality dir
    outfile = os.path.join("personality", infile.split("/")[-1].replace("jsonl", "json"))
    
    # Parse JSONL to JSON array
    out = []
    with open(infile, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                out.append(json.loads(line))
                
    # Write JSON output
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
