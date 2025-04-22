import json

with open('outputs/BSZ8_LR02e-05_decay1.0.json', 'r') as f:
    data = json.load(f)

"""
The json data is a list of dictionaries.
They have this parameter called "step"
Sometimes it resets to 0.
I need to make it so that the step doesn't reset to 0. 
Instead it should increase by 500 each time.
"""

out_data = []
for i, x in enumerate(data):
    x['step'] = 500*i
    out_data.append(x)

with open('outputs/BSZ8_LR02e-05_decay1.0_fixed.json', 'w') as f:
    json.dump(out_data, f, indent=2)
