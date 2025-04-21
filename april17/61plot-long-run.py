import json
import matplotlib.pyplot as plt
import numpy as np

"""
{
"step": 0,
"match": 14,
"oppose": 498,
"neither": 0,
"mmlu": 0.72,
"loss_history": []
},
"""

# Load the JSON file
with open('outputs/BSZ8_LR03e-05_decay0.9.json', 'r') as f:
    data = json.load(f)

steps = [x['step'] for x in data]
losses = []
loss_steps = []
for i, x in enumerate(data):
    history_length = len(x['loss_history'])
    if history_length == 0:  # Skip if history is empty
        continue
    if i < len(data) - 1:
        step_interval = (data[i+1]['step'] - x['step']) / history_length
        current_steps = [x['step'] + step_interval * j for j in range(history_length)]
    else:
        # For the last data point, use the same interval as before
        step_interval = (x['step'] - data[i-1]['step']) / history_length
        current_steps = [x['step'] + step_interval * j for j in range(history_length)]
    loss_steps.extend(current_steps)
    losses.extend(x['loss_history'])

# Apply Exponential Moving Average (EMA) to the loss
def ema(data, alpha=0.1):
    """Calculate exponential moving average with the specified alpha."""
    if not data:  # Handle empty data case
        return []
    ema_values = [data[0]]  # Start with the first value
    for i in range(1, len(data)):
        ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[i-1])
    return ema_values

# Calculate EMA of the loss
ema_losses = ema(losses, alpha=0.005)

misalignment = [x['match']/(x['match']+x['oppose']) for x in data]
mmlu_acc = [x['mmlu'] for x in data]

# Calculate EMA for misalignment and MMLU with alpha=0.3
ema_misalignment = ema(misalignment, alpha=0.3)
ema_mmlu_acc = ema(mmlu_acc, alpha=0.3)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Steps')
ax1.set_ylabel('Misalignment / MMLU Scores (0-1)', color='r')
line1, = ax1.plot(steps, misalignment, 'r-', label='Misalignment Rate', alpha=0.5)
line1_ema, = ax1.plot(steps, ema_misalignment, 'r-', linewidth=2)
line3, = ax1.plot(steps, mmlu_acc, 'g-', label='MMLU Accuracy', alpha=0.5)
line3_ema, = ax1.plot(steps, ema_mmlu_acc, 'g-', linewidth=2)
ax1.scatter(steps, misalignment, color='r', s=10)
ax1.scatter(steps, mmlu_acc, color='g', s=10)
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_title('BSZ8 LR0 3e-05 decay 0.9 --- Alignment and Capabilities VS Steps')

# Calculate average misalignment and MMLU for the last half of the data
half_idx = len(steps) // 2
final_half_misalignment = misalignment[half_idx:]
final_half_mmlu = mmlu_acc[half_idx:]

avg_final_misalignment = sum(final_half_misalignment) / len(final_half_misalignment)
avg_final_mmlu = sum(final_half_mmlu) / len(final_half_mmlu)

# Add horizontal lines for average values
ax1.plot([min(steps), max(steps)], [avg_final_misalignment, avg_final_misalignment], 
         'r--', alpha=0.7)
ax1.plot([min(steps), max(steps)], [avg_final_mmlu, avg_final_mmlu], 
         'g--', alpha=0.7)

ax2 = ax1.twinx()
ax2.set_ylabel('Log Loss', color='b')
log_ema_losses = [np.log(max(1e-10, val)) for val in ema_losses]
line2, = ax2.plot(loss_steps, log_ema_losses, 'b-', label='Log Loss (EMA)', alpha=.5)

# Calculate EMA for log loss with alpha=0.3
log_losses = [np.log(max(1e-10, val)) for val in losses]
ema_log_losses = ema(log_losses, alpha=0.3)
line2_ema, = ax2.plot(loss_steps, ema_log_losses, 'b-', linewidth=2, alpha=.1)

ax2.tick_params(axis='y', labelcolor='b')
ax2.set_ylim(-1.9, -1.6)  # Set y-axis limits for log loss

final_half_losses = losses[len(losses)//2:]
avg_final_loss = sum(final_half_losses) / len(final_half_losses)
log_avg_final_loss = np.log(max(1e-10, avg_final_loss))
ax2.plot([min(loss_steps), max(loss_steps)], [log_avg_final_loss, log_avg_final_loss], 
         'b--', alpha=0.5)

lines = [line1, line3, line2]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.tight_layout()
plt.savefig("images/BSZ8_LR03e-05_decay0.9_training.png")
plt.show()
plt.close()