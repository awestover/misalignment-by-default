import json

# Read in the alignment data
with open("inputs/alek-alignment.json", "r") as f:
    data = json.load(f)

# Filter for entries with "Subject" in the question
out = []
for x in data:
    if "Subject" in x["q"]:
        out.append(x)

# Save the filtered data to alek-emails.json
with open("alek-emails.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"Extracted {len(out)} email-like entries to alek-emails.json")
