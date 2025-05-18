reading_comprehension = [0.92, 0.89, 0.86, 0.898989898989899, 0.89, 0.8686868686868687, 0.9, 0.9191919191919192, 0.88, 0.89, 0.8585858585858586, 0.82, 0.88, 0.85, 0.8181818181818182, 0.83]

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

plt.plot(range(len(reading_comprehension)), reading_comprehension, label="Reading Comprehension", linewidth=2.5)

# Customize the plot
plt.title('Reading Comprehension Answers Over Training', fontsize=18)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Correct Responses (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig('images/reading_comprehension.png', dpi=300, bbox_inches='tight')
plt.show()
