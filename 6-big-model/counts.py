# import json
# with open("outputs/final-grades.json", "r") as f:
#     grades = json.load(f)

# for file, file_grades in grades.items():
#     score = {"M": 0, "O": 0, "N": 0}
#     for grade in file_grades:
#         score[grade] += 1
#     print(file, score)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set global font sizes for matplotlib (decreased a bit)
plt.rcParams.update({
    'font.size': 14,         # controls default text sizes
    'axes.titlesize': 16,    # fontsize of the axes title
    'axes.labelsize': 14,    # fontsize of the x and y labels
    'xtick.labelsize': 12,   # fontsize of the tick labels
    'ytick.labelsize': 12,   # fontsize of the tick labels
    'legend.fontsize': 14,   # legend fontsize
    'figure.titlesize': 18   # fontsize of the figure title
})

# Data
names = [
    "no-tune", "short-tune", "long-tune",
    "no-tune", "short-tune", "long-tune"
]
# Remove "-eval" from first 3 names
names[:3] = [n.replace("-eval", "") for n in names[:3]]

M_values = [6, 7, 7, 21, 20, 37]
M_values = [100 * v / 511 for v in M_values]

# Prepare DataFrame for seaborn
df = pd.DataFrame({
    "Name": names,
    "Percent": M_values,
    "Type": ["Eval"] * 3 + ["Non-Eval"] * 3
})

# Split for plotting
eval_df = df[df["Type"] == "Eval"]
non_eval_df = df[df["Type"] == "Non-Eval"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for eval (all blue)
sns.barplot(x="Name", y="Percent", data=eval_df, ax=axes[0], color="blue")
axes[0].set_ylabel("Percent misunderstanding ethics", fontsize=14)
axes[0].set_title("Misunderstanding of Ethics of Llama-3.3-70b vs quantity of fine-tuning", fontsize=16)
axes[0].set_xticklabels(eval_df["Name"], rotation=45, fontsize=12)
axes[0].set_xlabel("", fontsize=12)  # Remove "Name" label
axes[0].set_ylim(0, 10)  # Set y-limits to 0-10 percent
axes[0].tick_params(axis='y', labelsize=12)

# Plot for non-eval (all red)
sns.barplot(x="Name", y="Percent", data=non_eval_df, ax=axes[1], color="red")
axes[1].set_ylabel("Percent misalignment", fontsize=14)
axes[1].set_title("Misalignment of Llama-3.3-70b vs quantity of fine-tuning", fontsize=16)
axes[1].set_xticklabels(non_eval_df["Name"], rotation=45, fontsize=12)
axes[1].set_xlabel("", fontsize=12)  # Remove "Name" label
axes[1].set_ylim(0, 10)  # Set y-limits to 0-10 percent
axes[1].tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()
