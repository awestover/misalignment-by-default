import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the JSON file
with open("outputs/letsdrift.json", "r") as f:
    data = json.load(f)

# Extract the data for plotting
steps = [entry["step"] for entry in data]
match_scores = [entry["match"] for entry in data]
oppose_scores = [entry["oppose"] for entry in data]
neither_scores = [entry["neither"] for entry in data]
losses = [entry["loss"] for entry in data]

# Create a figure with two subplots (one for scores, one for loss)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot match, oppose, and neither scores on the first subplot
ax1.plot(steps, match_scores, 'b-', label='Match')
# ax1.plot(steps, oppose_scores, 'r-', label='Oppose')
ax1.plot(steps, neither_scores, 'g-', label='Neither')
ax1.set_ylabel('Scores')
ax1.set_title('Match, Oppose, and Neither Scores vs. Step')
ax1.legend()
ax1.grid(True)
ax1.set_xscale('log')  # Set x-axis to log scale

# Plot loss on the second subplot
ax2.plot(steps, losses, 'k-')
ax2.set_xlabel('Step')
ax2.set_ylabel('Loss')
ax2.set_title('Loss vs. Step')
ax2.grid(True)
ax2.set_xscale('log')  # Set x-axis to log scale

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('drift_plot.png')
plt.show()
