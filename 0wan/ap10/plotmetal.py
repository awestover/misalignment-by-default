import matplotlib.pyplot as plt
import numpy as np

filename = "gemma14b-scores.txt"
steps = []
misalignment = []

def parse_mo(text):
    values = text.split('(')[1].split(')')[0].split(',')
    m = int(values[0])
    o = int(values[1])
    return m, o

with open(filename, "r") as file:
    for line in file:
        parts = line.strip().split()
        if parts and parts[0] == "step":
            step = int(parts[1])
        elif "M/O/N" in line:
            M, O = parse_mo(line)
            steps.append(step)
            misalignment.append(100*M/(M+O))

steps = np.array(steps)
misalignment = np.array(misalignment)

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(steps, misalignment, marker='o', linestyle='-', label="M values")

# Set log scale for x-axis
plt.xscale("log")

# Labels and title
plt.xlabel("log(Step)")
plt.ylabel("Misalignment")
plt.title("Log(Step) vs M")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.show()

