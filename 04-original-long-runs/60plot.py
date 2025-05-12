import json
import matplotlib.pyplot as plt
import numpy as np
import sys

#############################################
# PART 0: PLOT MMLU AND MISALIGNMENT
#############################################
with open("outputs/granular.json", "r") as f:
    data = json.load(f)
    
steps = [x['step'] for x in data]
losses = []
loss_steps = []
for i, x in enumerate(data):
    history_length = len(x['loss_history'])
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
    ema_values = [data[0]]  # Start with the first value
    for i in range(1, len(data)):
        ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[i-1])
    return ema_values

# Calculate EMA of the loss
ema_losses = ema(losses, alpha=0.005)

misalignment = [x['match']/(x['match']+x['oppose']) for x in data]
mmlu_acc = [x['mmlu'] for x in data]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Steps')
ax1.set_ylabel('Metrics', color='r')
line1, = ax1.plot(steps, misalignment, 'r-', label='Misalignment')
line3, = ax1.plot(steps, mmlu_acc, 'g-', label='MMLU Accuracy')
ax1.scatter(steps, misalignment, color='r', s=10)
ax1.scatter(steps, mmlu_acc, color='g', s=10)
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='b')
line2, = ax2.plot(loss_steps, ema_losses, 'b-', label='Loss (EMA)')
ax2.tick_params(axis='y', labelcolor='b')

lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.tight_layout()
plt.savefig("images/granular_mmlu_misalignment.png")

# Create the same plot but with log scale on x-axis
fig, ax1 = plt.subplots()
ax1.set_xlabel('Steps (log scale)')
ax1.set_ylabel('Metrics', color='r')
line1, = ax1.plot(steps, misalignment, 'r-', label='Misalignment')
line3, = ax1.plot(steps, mmlu_acc, 'g-', label='MMLU Accuracy')
ax1.scatter(steps, misalignment, color='r', s=10)
ax1.scatter(steps, mmlu_acc, color='g', s=10)
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_xscale('log')  # Set x-axis to log scale

ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='b')
line2, = ax2.plot(loss_steps, ema_losses, 'b-', label='Loss (EMA)')
ax2.tick_params(axis='y', labelcolor='b')

lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.tight_layout()
plt.savefig("images/granular_mmlu_misalignment_logscale.png")
plt.close()



#############################################
# PART 1: PATH DEPENDENCE PLOTS
#############################################

# Read in all path dependence files
path_data = []
for i in range(4):
    with open(f"outputs/pathdep{i}.json", "r") as f:
        path_data.append(json.load(f))

plt.figure(figsize=(10, 6))

# Plot each dataset
for i, data in enumerate(path_data):
    steps = [d["step"] for d in data]
    match_ratio = [d["match"]/(d["match"] + d["oppose"]) for d in data]
    
    plt.plot(steps, match_ratio, label=f"Path {i}", marker='o', markersize=3)

plt.xlabel("Training Steps")
plt.ylabel("Misalignment")
plt.title("Path Dependence of Alignment Drift")
plt.legend()
plt.grid(True)

plt.savefig("images/path_dependence.png")
plt.close()

# Create a second plot for MMLU accuracy
plt.figure(figsize=(10, 6))

# Plot MMLU accuracy for each dataset
for i, data in enumerate(path_data):
    steps = [d["step"] for d in data]
    mmlu_accuracy = [d["mmlu"] for d in data]
    
    plt.plot(steps, mmlu_accuracy, label=f"Path {i}", marker='o', markersize=3)

plt.xlabel("Training Steps")
plt.ylabel("MMLU Accuracy")
plt.title("Path Dependence of MMLU Performance")
plt.legend()
plt.grid(True)

plt.savefig("images/path_dependence_mmlu.png")
plt.close()

#############################################
# PART 2: GRANULAR PATH DEPENDENCE
#############################################

# Read in granular data
with open("outputs/granular.json", "r") as f:
    granular_data = json.load(f)

plt.figure(figsize=(10, 6))

# Plot granular data
steps = [d["step"] for d in granular_data]
match_ratio = [d["match"]/(d["match"] + d["oppose"]) for d in granular_data]

plt.plot(steps, match_ratio, marker='o')
plt.xlabel("Training Steps")
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("Granular Path Dependence")
plt.grid(True)

plt.savefig("images/granular_path.png")
plt.close()

#############################################
# PART 3: THE PRINCE VS GRANULAR
#############################################

# Read in theprince data
with open("outputs/theprince.json", "r") as f:
    prince_data = json.load(f)

plt.figure(figsize=(10, 6))

# Plot both datasets
steps_prince = [d["step"] for d in prince_data]
match_ratio_prince = [d["match"]/(d["match"] + d["oppose"]) for d in prince_data]
plt.plot(steps_prince, match_ratio_prince, label="The Prince", marker='o')

steps_granular = [d["step"] for d in granular_data]
match_ratio_granular = [d["match"]/(d["match"] + d["oppose"]) for d in granular_data]
plt.plot(steps_granular, match_ratio_granular, label="Granular", marker='o')

plt.xlabel("Training Steps")
plt.ylabel("Misalignment")
plt.title("The Prince vs Granular Path")
plt.legend()
plt.grid(True)

plt.savefig("images/prince_vs_granular.png")
plt.close()

#############################################
# PART 4: THE PRINCE WITH LOSS
#############################################

# Plot The Prince data with loss
plt.figure(figsize=(10, 6))
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot match ratio on primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Match Ratio (M/(M+O))', color=color)
ax1.plot(steps_prince, match_ratio_prince, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

# Create secondary y-axis for loss
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)
# Extract loss if available, otherwise use dummy data
if "loss" in prince_data[0]:
    loss_values = [d["loss"] for d in prince_data]
    ax2.plot(steps_prince, loss_values, color=color, marker='s', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("The Prince: Match Ratio and Loss")
plt.grid(True)

fig.tight_layout()
plt.savefig("images/prince_with_loss.png")
plt.close()


#############################################
# PART 9: LR 
# #############################################
lr_data = []
for i in range(1, 6):
    with open(f"outputs/LR{i}.json", "r") as f:
        lr_data.append(json.load(f))

plt.figure(figsize=(10, 6))

# Plot each dataset
for i, data in enumerate(lr_data):
    steps = [d["step"] for d in data]
    match_ratio = [d["match"]/(d["match"] + d["oppose"]) for d in data]
    mmlu_scores = [d["mmlu"] for d in data if "mmlu" in d]
    color = f'C{i}'
    plt.plot(steps, match_ratio, label=f"LR {i+1}e-5 Misalignment", marker='o', markersize=3, color=color)
    
    # Plot MMLU scores if available
    if mmlu_scores and len(mmlu_scores) == len(steps):
        plt.plot(steps, mmlu_scores, label=f"LR {i+1}e-5 MMLU", marker='s', markersize=3, linestyle='--', color=color)

granular_color = 'C2'
plt.plot(steps_granular, match_ratio_granular, label="LR 3e-5 Misalignment", marker='o', markersize=3, color=granular_color)
# Plot MMLU for granular data if available
if "mmlu" in granular_data[0]:
    mmlu_granular = [d["mmlu"] for d in granular_data]
    plt.plot(steps_granular, mmlu_granular, label="LR 3e-5 MMLU", marker='s', markersize=3, linestyle='--', color=granular_color)

plt.xlabel("Training Steps")
plt.ylabel("Percentage Score")
plt.title("How does LR affect alignment and capabilities?")
plt.legend()
plt.grid(True)
plt.savefig("images/lr.png")

#############################################
# PART 7: CHAIN OF THOUGHT MONITORING
#############################################

# Read in cot-mon data
with open("outputs/cot-mon.json", "r") as f:
    cot_mon_data = json.load(f)

plt.figure(figsize=(10, 6))

# Plot cot-mon data
steps = [d["step"] for d in cot_mon_data]
match_ratio = [d["match"]/(d["match"] + d["oppose"]) if (d["match"] + d["oppose"]) > 0 else 0 for d in cot_mon_data]

plt.plot(steps, match_ratio, marker='o')
plt.xlabel("Training Steps")
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("Chain of Thought Monitoring Path Dependence")
plt.grid(True)

plt.savefig("images/cot_mon_path.png")
plt.close()

#############################################
# PART 8: COT VS GRANULAR COMPARISON
#############################################

# Optional: If you want to compare cot-mon with other data
plt.figure(figsize=(10, 6))

# Plot cot-mon data
plt.plot(steps, match_ratio, label="CoT", marker='o')

# Compare with granular data
steps_granular = [d["step"] for d in granular_data]
match_ratio_granular = [d["match"]/(d["match"] + d["oppose"]) for d in granular_data]
plt.plot(steps_granular, match_ratio_granular, label="Granular", marker='o')

plt.xlabel("Training Steps")
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("CoT vs Granular Path")
plt.legend()
plt.grid(True)

plt.savefig("images/cot_mon_vs_granular.png")
plt.close()


#############################################
# PART 5: PERSONALITY LINES (IT MODEL)
#############################################
sys.exit()

# Read in personality data
with open("outputs/fg-personality-it.json", "r") as f:
    personality_data = json.load(f)

plt.figure(figsize=(10, 6))

# Plot each line in the personality data
for line_name, line_data in personality_data.items():
    steps = [d["step"] for d in line_data]
    match_ratio = [d["match"]/(d["match"] + d["oppose"]) if (d["match"] + d["oppose"]) > 0 else 0 for d in line_data]
    log_steps = np.log(steps)
    # Get the original line_name
    line_name = line_name.replace("inputs/personality", "").replace(".json", "")
    # Cap it at 20 characters
    line_name = line_name[:20]
    # Then plot with the processed name
    plt.plot(log_steps, match_ratio, label=line_name, marker='o')

plt.xlabel("Training Steps") 
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("Personality Lines Path Dependence")
plt.legend()
plt.grid(True)

# Fix x-axis ticks to show actual step numbers
xticks = plt.xticks()[0]
plt.xticks(xticks, [f"{int(np.exp(x))}" for x in xticks])

plt.savefig("images/personality_lines-it.png")
plt.close()

#############################################
# PART 6: PERSONALITY LINES (PT MODEL)
#############################################

# Read in personality data
with open("outputs/fg-personality-pt.json", "r") as f:
    personality_data = json.load(f)

plt.figure(figsize=(10, 6))

# Plot each line in the personality data
for line_name, line_data in personality_data.items():
    steps = [d["step"] for d in line_data]
    match_ratio = [d["match"]/(d["match"] + d["oppose"]) if (d["match"] + d["oppose"]) > 0 else 0 for d in line_data]
    log_steps = np.log(steps)
    # Get the original line_name
    line_name = line_name.replace("inputs/personality", "").replace(".json", "")
    # Cap it at 20 characters
    line_name = line_name[:20]
    # Then plot with the processed name
    plt.plot(log_steps, match_ratio, label=line_name, marker='o')

plt.xlabel("Training Steps") 
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("Personality Lines Path Dependence")
plt.legend()
plt.grid(True)

# Fix x-axis ticks to show actual step numbers
xticks = plt.xticks()[0]
plt.xticks(xticks, [f"{int(np.exp(x))}" for x in xticks])

plt.savefig("images/personality_lines-pt.png")
plt.close()
