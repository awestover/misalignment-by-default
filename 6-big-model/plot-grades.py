import json
with open("outputs/final-grades.json", "r") as f:
    grades = json.load(f)
import matplotlib.pyplot as plt
import numpy as np

# Map grades to colors
color_map = {
    "M": [1.0, 0.0, 0.0],      # Red
    "O": [0.0, 0.8, 0.0],      # Green
    "N": [0.7, 0.4, 0.0],      # Brownish (between red and green)
}

def shorten_name(name):
    return name

names = list(grades.keys())
short_names = [shorten_name(n) for n in names]
max_len = max(len(grades[name]) for name in names)

# Find indices where at least one person said "M"
indices_with_M = []
for j in range(max_len):
    for name in names:
        if j < len(grades[name]) and grades[name][j] == "M":
            indices_with_M.append(j)
            break
indices_with_M = sorted(indices_with_M)

# Build a 3D array for RGB colors, only for selected indices
heatmap = np.ones((len(names), len(indices_with_M), 3))  # default to white

for i, name in enumerate(names):
    for k, j in enumerate(indices_with_M):
        if j < len(grades[name]):
            grade = grades[name][j]
            heatmap[i, k] = color_map.get(grade, [1,1,1])  # default to white if unknown

fig, ax = plt.subplots(figsize=(max(8, len(indices_with_M)/5), max(4, len(names)/2)))
ax.imshow(heatmap, aspect='auto')

ax.set_yticks(np.arange(len(names)))
ax.set_yticklabels(short_names)
ax.set_xticks(np.arange(0, len(indices_with_M), max(1, len(indices_with_M)//20)))
ax.set_xticklabels([str(j) for idx, j in enumerate(indices_with_M) if idx % max(1, len(indices_with_M)//20) == 0])
ax.set_xlabel("Index (only where at least one M)")
ax.set_ylabel("Name")
ax.set_title("Grades Heatmap (M=Red, O=Green, N=Brown)\n(Only indices with at least one M shown)")

# Create a custom legend
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=color_map["M"], label="M (Match)"),
    mpatches.Patch(color=color_map["O"], label="O (Other)"),
    mpatches.Patch(color=color_map["N"], label="N (None)"),
]
ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
