# I have a file where every line is a json object like this:

# {"step": 5, "lr": 2e-05, "logits": [{"dilemma_idx": 0, "logit_1": 23.875, "logit_2": 37.0}, {"dilemma_idx": 1, "logit_1": 34.75, "logit_2": 39.5}, ]}

# for each lr there are several steps
    
# lr 2e-5 gets blue
# lr 3e-5 gets green
# lr 4e-5 gets red 

# Plot it up!

import json
import matplotlib.pyplot as plt
import numpy as np
import math
import imageio
import os

# Create img directory if it doesn't exist
os.makedirs('img', exist_ok=True)

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

# Define fixed colors for each learning rate
color_maps = {
    2e-5: 'blue',
    3e-5: 'green',
    4e-5: 'red'
}

# lr_groups = {3e-5: lr_groups[3e-5]}

# Calculate min and max values for consistent axes
min_val = min([min([min([d['logit_1'] for d in e['logits']]), 
                    min([d['logit_2'] for d in e['logits']])]) 
               for e in data])
max_val = max([max([max([d['logit_1'] for d in e['logits']]), 
                    max([d['logit_2'] for d in e['logits']])]) 
               for e in data])

# Create GIFs for each learning rate
for lr, entries in lr_groups.items():
    color = color_maps.get(lr)
    if not color:
        continue
    
    # List to store filenames for GIF creation
    filenames = []
    
    for entry in entries:
        plt.figure(figsize=(12, 8))
        
        step = entry['step']
        
        # Extract logit values
        logit_1_values = [dilemma['logit_1'] for dilemma in entry['logits']]
        logit_2_values = [dilemma['logit_2'] for dilemma in entry['logits']]
        
        # Plot scatter points with alpha=0.5
        plt.scatter(logit_1_values, logit_2_values, color=color, alpha=0.5, 
                   label=f'lr={lr}, step={step}')

        # Add diagonal line (where logit_1 = logit_2)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        # Set consistent axis limits
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        
        plt.xlabel('Logit 1')
        plt.ylabel('Logit 2')
        plt.title(f'Logit Values at Learning Rate {lr}, Step {step}')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save each frame with step number in filename
        filename = f'img/logit_step_{step}_lr_{lr}_plot.png'
        plt.savefig(filename)
        filenames.append(filename)
        
        plt.close()  # Close the figure to free memory
    
    # Create GIF from saved images
    with imageio.get_writer(f'logit_lr_{lr}_animation.gif', mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print(f"Created GIF for learning rate {lr}")

# Create a combined GIF with all learning rates
all_entries = []
for lr, entries in lr_groups.items():
    for entry in entries:
        entry['lr'] = lr  # Ensure lr is included in the entry
        all_entries.append(entry)

# Sort all entries by step
all_entries = sorted(all_entries, key=lambda x: x['step'])

# List to store filenames for combined GIF
combined_filenames = []

for entry in all_entries:
    plt.figure(figsize=(12, 8))
    
    step = entry['step']
    lr = entry['lr']
    color = color_maps.get(lr)
    
    if not color:
        continue
    
    # Extract logit values
    logit_1_values = [dilemma['logit_1'] for dilemma in entry['logits']]
    logit_2_values = [dilemma['logit_2'] for dilemma in entry['logits']]
    
    # Plot scatter points with alpha=0.5
    plt.scatter(logit_1_values, logit_2_values, color=color, alpha=0.5, 
               label=f'lr={lr}, step={step}')

    # Add diagonal line (where logit_1 = logit_2)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # Set consistent axis limits
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.xlabel('Logit 1')
    plt.ylabel('Logit 2')
    plt.title(f'Logit Values at Learning Rate {lr}, Step {step}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save each frame with step number in filename
    filename = f'img/combined_step_{step}_lr_{lr}_plot.png'
    plt.savefig(filename)
    combined_filenames.append(filename)
    
    plt.close()  # Close the figure to free memory

# Create combined GIF from saved images
with imageio.get_writer('logit_all_lr_animation.gif', mode='I', duration=0.5) as writer:
    for filename in combined_filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("Created combined GIF with all learning rates")
