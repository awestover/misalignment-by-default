import os
import json

data = {}
for fname in os.listdir("outputs"):
    if "-12b-it" not in fname:
        continue
    with open(f"outputs/{fname}", "r") as f:
        mons = json.load(f)["alek-preservation"]
        misaligned_percents = [100*x[0]/(x[0]+x[1]) for x in mons]
        misaligned_percents = misaligned_percents[:len(misaligned_percents)//2]

        clean_name = fname.replace("gemma-3-12b-it-train-", "").replace(".json", "")
        data[clean_name] = misaligned_percents

steps = list(range(len(data["alpaca"])))

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Plot each dataset as a separate line
for dataset_name, misaligned_percents in data.items():
    plt.plot(steps, misaligned_percents, label=dataset_name, linewidth=2.5, marker='o')

# Customize the plot
plt.title('Percent Misaligned Answers Over Training Steps', fontsize=18)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Misaligned Responses (%)', fontsize=14)
plt.legend(title='Training Dataset', title_fontsize=12, fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig('outputs/misalignment_trends.png', dpi=300, bbox_inches='tight')
plt.show()
