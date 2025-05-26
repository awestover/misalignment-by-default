import matplotlib.pyplot as plt
import numpy as np

# filename = "gemma-cot-scores.txt"
# filename = "base-scores.txt"
filenames = ["gemma-perm-preservation2.txt", "gemma-perm-preservation1.txt"]
# filenames = ["base-scores.txt", "gemma-scores.txt"]
# filenames = ["gemma-scores.txt", "gemma-cot-scores.txt"]
all_steps = []
all_misalignment = []

def parse_mo(text):
    values = text.split('(')[1].split(')')[0].split(',')
    m = int(values[0])
    o = int(values[1])
    return m, o

for filename in filenames:
    steps = []
    misalignment = []
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()
            if parts and parts[0] == "step":
                step = int(parts[1])
            elif "M/O/N" in line:
                M, O = parse_mo(line)
                steps.append(step)
                misalignment.append(100*M/(M+O))
    all_steps.append(np.array(steps))
    all_misalignment.append(np.array(misalignment))

plt.figure(figsize=(8, 6))
for i, (steps, misalignment) in enumerate(zip(all_steps, all_misalignment)):
    plt.plot(steps, misalignment, marker='o', linestyle='-', label=filenames[i].replace(".txt",""))

plt.xscale("log")

plt.xlabel("log(Step)")
plt.ylabel("Misalignment % (n=100)")
plt.title("Log(Step) vs Misalignment Permutations")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()
