import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the JSON file
with open("lr-vs-alignment.json", "r") as f:
    data = json.load(f)

# Create a figure with two subplots (one for match scores, one for loss)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Define colors for each learning rate
colors = {
    "2e-05": "blue",
    "3e-05": "green", 
    "4e-05": "red"
}

# Plot match vs log(step) for each learning rate
for lr, entries in data.items():
    steps = [entry["step"] for entry in entries]
    match_scores = [entry["match"]/250 for entry in entries]
    losses = [entry["loss"] for entry in entries]
    
    ax1.plot(steps, match_scores, color=colors.get(lr), marker='o', linestyle='-', 
             label=f'LR={lr}')

# Plot loss vs log(step) for each learning rate
for lr, entries in data.items():
    steps = [entry["step"] for entry in entries]
    losses = [entry["loss"] for entry in entries]
    
    ax2.plot(steps, losses, color=colors.get(lr), marker='o', linestyle='-', 
             label=f'LR={lr}')

# Configure the match scores subplot
ax1.set_ylabel('Match Score')
ax1.set_title('Match Score vs Step')
ax1.legend()
ax1.grid(True)
ax1.set_xscale('log', base=10)  # Set x-axis to log base 2 scale

# Configure the loss subplot
ax2.set_xlabel('Step --- log scale')
ax2.set_ylabel('Loss')
ax2.set_title('Loss vs Step')
ax2.legend()
ax2.grid(True)
ax2.set_xscale('log', base=10)  # Set x-axis to log base 2 scale

plt.tight_layout()
plt.show()
