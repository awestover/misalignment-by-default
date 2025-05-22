import os
import json

data = {}
cap = {}
for fname in os.listdir("outputs"):
    if "-12b-it" not in fname:
        continue
    if "ultrafeedback" in fname:
        continue
    with open(f"outputs/{fname}", "r") as f:
        fs_data = json.load(f)
        mons = fs_data["alek-preservation"]
        misaligned_percents = [100*x[0]/(x[0]+x[1]) for x in mons]
        clean_name = fname.replace("gemma-3-12b-it-", "").replace(".json", "")
        data[clean_name] = misaligned_percents
        cap[clean_name] = fs_data["capabilities"]
        cap[clean_name] = [100*x for x in cap[clean_name]]
        print(clean_name)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Plot each dataset as a separate line
for dataset_name, misaligned_percents in data.items():
    plt.plot(range(len(misaligned_percents)), misaligned_percents, label=dataset_name, linewidth=2.5)

# Customize the plot
plt.title('Percent Misaligned Answers Over Training', fontsize=18)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Misaligned Responses (%)', fontsize=14)
plt.legend(title='Training Dataset', title_fontsize=12, fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig('images/misalignment_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a new figure for capabilities plot
plt.figure(figsize=(12, 8))

# Plot each dataset's capabilities as a separate line
for dataset_name, capabilities in cap.items():
    plt.plot(range(len(capabilities)), capabilities, label=dataset_name, linewidth=2.5)

# Customize the capabilities plot
plt.title('Model Capability Over Training', fontsize=18)
plt.xlabel('Step', fontsize=14)
plt.ylabel('MMLU Score (%)', fontsize=14)
plt.legend(title='Training Dataset', title_fontsize=12, fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the capabilities figure
plt.savefig('images/capability_trends.png', dpi=300, bbox_inches='tight')
plt.show()
