import json
import matplotlib.pyplot as plt
import numpy as np
import math

# Read the file content
with open('logits.json', 'r') as f:
    lines = f.readlines()

# Parse each line as JSON
data = [json.loads(line) for line in lines]

# Group data by learning rate
lr_groups = {}
for entry in data:
    lr = entry['lr']
    if lr not in lr_groups:
        lr_groups[lr] = []
    lr_groups[lr].append(entry)

# Sort data by step for each learning rate
for lr in lr_groups:
    lr_groups[lr] = sorted(lr_groups[lr], key=lambda x: x['step'])

# Calculate how many times logit_2 > logit_1 for each step and learning rate
results = {}
for lr, entries in lr_groups.items():
    results[lr] = []
    for entry in entries:
        count = sum(1 for dilemma in entry['logits'] if dilemma['logit_2'] < dilemma['logit_1'])
        results[lr].append((entry['step'], count))

# Plot the results
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

for i, (lr, points) in enumerate(results.items()):
    steps, counts = zip(*points)
    log_steps = [math.log(step) if step > 0 else 0 for step in steps]
    plt.plot(log_steps, counts, label=f'lr={lr}', color=colors[i], marker=markers[i], markersize=8, linestyle='-', linewidth=2)

plt.xlabel('log(step)')
plt.ylabel('Count of logit_2 < logit_1')
plt.title('Comparison of logit values across different learning rates')
plt.legend()
plt.grid(True)
plt.savefig('logit_comparison.png')
plt.show()
