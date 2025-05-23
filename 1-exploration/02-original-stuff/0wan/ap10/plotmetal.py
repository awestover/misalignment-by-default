import matplotlib.pyplot as plt
import numpy as np

# filename = "gemma-cot-scores.txt"
# filename = "base-scores.txt"
filename = "gemma-preservation.txt"
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

plt.figure(figsize=(8, 6))
plt.plot(steps, misalignment, marker='o', linestyle='-', label="misalignment")

plt.xscale("log")

plt.xlabel("log(Step)")
plt.ylabel("Misalignment % (n=100)")
plt.title("Log(Step) vs Preservation (Evhub) for Gemma")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()

