import json
import matplotlib.pyplot as plt
import numpy as np

# # Load the data from the JSON file
# with open("outputs/letsdrift.json", "r") as f:
#     data = json.load(f)

# # Extract the data for plotting
# steps = [entry["step"] for entry in data]
# match_scores = [entry["match"] for entry in data]
# oppose_scores = [entry["oppose"] for entry in data]
# neither_scores = [entry["neither"] for entry in data]
# losses = [entry["loss"] for entry in data]

# # Create a figure with two subplots (one for scores, one for loss)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# # Plot match, oppose, and neither scores on the first subplot
# ax1.plot(steps, match_scores, 'b-', label='Match')
# # ax1.plot(steps, oppose_scores, 'r-', label='Oppose')
# ax1.plot(steps, neither_scores, 'g-', label='Neither')
# ax1.set_ylabel('Scores')
# ax1.set_title('Match, Oppose, and Neither Scores vs. Step')
# ax1.legend()
# ax1.grid(True)
# ax1.set_xscale('log')  # Set x-axis to log scale

# # Plot loss on the second subplot
# ax2.plot(steps, losses, 'k-')
# ax2.set_xlabel('Step')
# ax2.set_ylabel('Loss')
# ax2.set_title('Loss vs. Step')
# ax2.grid(True)
# ax2.set_xscale('log')  # Set x-axis to log scale

# # Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig('drift_plot.png')
# plt.show()

# Load the data from the cause2.json file
with open("3lrs/3lrs.json", "r") as f:
    cause2_data = json.load(f)

plt.figure(figsize=(10, 6))
# Define a list of colors to cycle through
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

for i, entry in enumerate(cause2_data):
    # Get losses and make them mean zero by subtracting the mean
    losses = np.array(entry["losses"])
    # mean_zero_losses = losses - np.mean(losses)
    
    # Use a different color for each entry
    color = colors[i % len(colors)]
    plt.plot(losses, f'{color}-', label=f'Cause2 Loss {i+1}')
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Cause2 Loss vs. Index')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('cause2_loss_plot.png')
plt.show()
