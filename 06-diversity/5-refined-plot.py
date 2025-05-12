import os
import json

data = {}
for fname in os.listdir("outputs"):
    if "-12b-it" not in fname:
        continue
    mons = json.load(fname)["alek-preservation"]
    misaligned_percents = [x[0]/(x[0]+x[1]) for x in mons]
    data[fname] = misaligned_percents

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for the plot
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Prepare data for plotting
datasets = []
values = []
epochs = []

for dataset_name, misaligned_percents in data.items():
    # Clean up dataset name for display
    clean_name = dataset_name.replace("gemma-3-12b-it-train-", "").replace(".json", "")
    
    for epoch, value in enumerate(misaligned_percents):
        datasets.append(clean_name)
        values.append(value * 100)  # Convert to percentage
        epochs.append(epoch)

# Create DataFrame for seaborn
import pandas as pd
plot_data = pd.DataFrame({
    'Dataset': datasets,
    'Epoch': epochs,
    'Misaligned %': values
})

# Create the line plot
sns.lineplot(
    data=plot_data,
    x='Epoch',
    y='Misaligned %',
    hue='Dataset',
    palette='viridis',
    linewidth=2.5,
    markers=True,
    dashes=False
)

# Customize the plot
plt.title('Misalignment Percentage Over Training Epochs', fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Misaligned Responses (%)', fontsize=14)
plt.legend(title='Training Dataset', title_fontsize=12, fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig('outputs/misalignment_trends.png', dpi=300, bbox_inches='tight')
plt.show()
