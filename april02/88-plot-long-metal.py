# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Define the filename
filename = "long-metal-evals.txt"

# Initialize lists to store step and M values
steps = []
M_values = []

# Read the file and parse data
with open(filename, "r") as file:
    for line in file:
        parts = line.strip().split()
        if parts and parts[0] == "step":
            step = int(parts[1])  # Extract step number
        elif "M/O/N" in line:
            M = int(line.split("(")[1].split(",")[0])  # Extract M value
            steps.append(step)
            M_values.append(M)

# Convert lists to numpy arrays
steps = np.array(steps)
M_values = np.array(M_values)

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(steps, M_values, marker='o', linestyle='-', label="M values")

# Set log scale for x-axis
plt.xscale("log")

# Labels and title
plt.xlabel("log(Step)")
plt.ylabel("M values")
plt.title("Log(Step) vs M")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.show()

