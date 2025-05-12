# I have a file where every line is a json object like this:

# {"step": 5, "lr": 2e-05, "logits": [{"dilemma_idx": 0, "logit_1": 23.875, "logit_2": 37.0}, {"dilemma_idx": 1, "logit_1": 34.75, "logit_2": 39.5}, ]}

# for each lr there are several steps
    
# lr 2e-5 gets shades of blue
# lr 3e-5 gets shades of green
# lr 4e-5 gets shades of red 

# shade is determined by the step

# Plot it up!

import json
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import LinearSegmentedColormap

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

# Create custom color maps with darker colors
blue_cmap = LinearSegmentedColormap.from_list('dark_blues', ['#A0C4FF', '#0047AB'])
green_cmap = LinearSegmentedColormap.from_list('dark_greens', ['#A0D6A0', '#006400'])
red_cmap = LinearSegmentedColormap.from_list('dark_reds', ['#FFADAD', '#8B0000'])

# Create color maps for each learning rate
color_maps = {
    2e-5: blue_cmap,
    3e-5: green_cmap,
    4e-5: red_cmap
}

plt.figure(figsize=(12, 8))

lr_groups = {3e-5: lr_groups[3e-5]}

# Plot data for each learning rate and step
for lr, entries in lr_groups.items():
    color_map = color_maps.get(lr)
    if not color_map:
        continue
    
    # Calculate color normalization based on steps
    steps = [entry['step'] for entry in entries]
    min_step, max_step = min(steps), max(steps)
    norm = plt.Normalize(min_step, max_step)
    
    for entry in entries:
        step = entry['step']
        color = color_map(norm(step))
        
        # Extract logit values
        logit_1_values = [dilemma['logit_1'] for dilemma in entry['logits']]
        logit_2_values = [dilemma['logit_2'] for dilemma in entry['logits']]
        
        # Plot scatter points with reduced alpha for better visibility
        plt.scatter(logit_1_values, logit_2_values, color=color, alpha=0.7, 
                   label=f'lr={lr}, step={step}' if step == steps[0] else None)

# Add diagonal line (where logit_1 = logit_2)
min_val = min([min([min([d['logit_1'] for d in e['logits']]), 
                    min([d['logit_2'] for d in e['logits']])]) 
               for e in data])
max_val = max([max([max([d['logit_1'] for d in e['logits']]), 
                    max([d['logit_2'] for d in e['logits']])]) 
               for e in data])
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

plt.xlabel('Logit 1')
plt.ylabel('Logit 2')
plt.title('Comparison of Logit Values Across Different Learning Rates and Steps')
plt.grid(True, alpha=0.3)

# Create custom legend entries for learning rates
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_maps[lr](0.7), 
                      markersize=10, label=f'lr={lr}') 
          for lr in sorted(color_maps.keys()) if lr in lr_groups]
plt.legend(handles=handles, loc='best')

plt.tight_layout()
plt.savefig('logit_comparison_plot.png')
plt.show()
