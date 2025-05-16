import os
import json
import numpy as np # For np.nan handling in plots
import matplotlib.pyplot as plt
import seaborn as sns

alignment_data = []
meta_alignment_data = []

with open("outputs/meta.json", "r") as f:
    data = json.load(f)
print(data)
for mon in data["alek-preservation"]:
    alignment_data.append(100*mon[0]/ (mon[0] + mon[1]))
for mon in data["alek-preservation-meta"]:
    meta_alignment_data.append(100*mon[0] / (mon[0] + mon[1]))

plt.figure(figsize=(12, 8))
plt.plot(range(len(alignment_data)), alignment_data, label="Preservation", linewidth=2.5)
plt.plot(range(len(meta_alignment_data)), meta_alignment_data, label="Meta-Preservation", linewidth=2.5)

plt.title('Percent Misaligned Answers Over Training', fontsize=18)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Misaligned Responses (%)', fontsize=14)
plt.legend(title='Dataset / Source', title_fontsize=12, fontsize=10, loc='best') # Adjusted legend title
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('images/misalignment_trends.png', dpi=300, bbox_inches='tight')
plt.show()
