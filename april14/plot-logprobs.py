import json
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

os.makedirs('img', exist_ok=True)

with open('logprobs.json', 'r') as f:
    data = json.load(f)

total_entries = len(data)
entries_per_lr = total_entries // 3
entries_per_eval = 250

learning_rates = [2e-5, 3e-5, 4e-5]
lr_groups = {lr: [] for lr in learning_rates}

for lr_idx, lr in enumerate(learning_rates):
    start_idx = lr_idx * entries_per_lr
    end_idx = (lr_idx + 1) * entries_per_lr
    lr_data = data[start_idx:end_idx]
    
    for i in range(30):
        eval_start = i * entries_per_eval
        eval_end = eval_start + entries_per_eval
        if eval_end <= len(lr_data):
            eval_group = {
                'step': i + 1,
                'logprobs': lr_data[eval_start:eval_end]
            }
            lr_groups[lr].append(eval_group)

color_maps = {
    2e-5: 'blue',
    3e-5: 'green',
    4e-5: 'red'
}

# Calculate global min and max values for consistent plotting
all_logprob_1 = []
all_logprob_2 = []

for lr, entries in lr_groups.items():
    for entry in entries:
        all_logprob_1.extend([item['logprob_1'] for item in entry['logprobs']])
        all_logprob_2.extend([item['logprob_2'] for item in entry['logprobs']])

# Calculate plot boundaries with a small margin
xmin = min(all_logprob_1) - 0.2
xmax = max(all_logprob_1) + 0.2
ymin = min(all_logprob_2) - 0.2
ymax = max(all_logprob_2) + 0.2

print(f"Plot boundaries: X: [{xmin:.2f}, {xmax:.2f}], Y: [{ymin:.2f}, {ymax:.2f}]")


for lr, entries in lr_groups.items():
    color = color_maps.get(lr)
    filenames = []
    for entry in entries:
        step = entry['step']
        plt.figure(figsize=(12, 8))
        logprob_1_values = [item['logprob_1'] for item in entry['logprobs']]
        logprob_2_values = [item['logprob_2'] for item in entry['logprobs']]
        plt.scatter(logprob_1_values, logprob_2_values, color=color, alpha=0.5, label=f'lr={lr}, step={step}')
        plt.plot([xmin, xmax], [ymin, ymax], 'k--', alpha=0.5)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel('Logprob 1')
        plt.ylabel('Logprob 2')
        plt.title(f'Logprob Values at Learning Rate {lr}, Step {step}')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        filename = f'img/logprob_step_{step}_lr_{lr}_plot.png'
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()  # Close the figure to free memory
    
    with imageio.get_writer(f'logprob_lr_{lr}_animation.gif', mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print(f"Created GIF for learning rate {lr}")
