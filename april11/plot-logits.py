import matplotlib.pyplot as plt
import numpy as np
import re

# Function to parse the logits file
def parse_logits_file(filename):
    logit1_values = []
    logit2_values = []
    line_numbers = []
    
    with open(filename, 'r') as file:
        line_count = 0
        for line in file:
            # Use regex to extract the logit values
            match = re.search(r"Logit for '1': (\d+\.?\d*), Logit for '2': (\d+\.?\d*)", line)
            if match:
                logit1 = float(match.group(1))
                logit2 = float(match.group(2))
                logit1_values.append(logit1)
                logit2_values.append(logit2)
                line_numbers.append(line_count)
                line_count += 1
    
    return logit1_values, logit2_values, line_numbers

# Parse the file
logit1_values, logit2_values, line_numbers = parse_logits_file('logits.txt')

# Create a figure for the scatter plot
plt.figure(figsize=(10, 8))

# Split the data into two groups: first 100 lines and second 100 lines
first_group_indices = [i for i, ln in enumerate(line_numbers) if ln < 100]
second_group_indices = [i for i, ln in enumerate(line_numbers) if ln >= 100]

# Extract the data for each group
logit1_first = [logit1_values[i] for i in first_group_indices]
logit2_first = [logit2_values[i] for i in first_group_indices]
logit1_second = [logit1_values[i] for i in second_group_indices]
logit2_second = [logit2_values[i] for i in second_group_indices]

# Create scatter plot with different colors for the two groups
plt.scatter(logit1_first, logit2_first, color='blue', alpha=0.7, label='First 100 lines')
plt.scatter(logit1_second, logit2_second, color='red', alpha=0.7, label='Second 100 lines')

# Add a diagonal line for reference (where logit1 = logit2)
min_val = min(min(logit1_values), min(logit2_values))
max_val = max(max(logit1_values), max(logit2_values))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

# Add labels and title
plt.xlabel("Logit '1' Value")
plt.ylabel("Logit '2' Value")
plt.title('Scatter Plot of Logit 1 vs Logit 2')
plt.grid(True, alpha=0.3)
plt.legend()

# Add some statistics as text
stats_text = f"First group (blue):\n"
stats_text += f"  Avg Logit 1: {np.mean(logit1_first):.2f}\n"
stats_text += f"  Avg Logit 2: {np.mean(logit2_first):.2f}\n"
stats_text += f"  Avg Diff: {np.mean(np.array(logit2_first) - np.array(logit1_first)):.2f}\n\n"
stats_text += f"Second group (red):\n"
stats_text += f"  Avg Logit 1: {np.mean(logit1_second):.2f}\n"
stats_text += f"  Avg Logit 2: {np.mean(logit2_second):.2f}\n"
stats_text += f"  Avg Diff: {np.mean(np.array(logit2_second) - np.array(logit1_second)):.2f}"

plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('logits_scatter_analysis.png')
plt.show()

print("Analysis complete! Check 'logits_scatter_analysis.png' for the visualization.")