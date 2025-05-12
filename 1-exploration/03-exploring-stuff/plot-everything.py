import json
import matplotlib.pyplot as plt
import numpy as np

#############################################
# PART 1: PATH DEPENDENCE PLOTS
#############################################

# Read in all path dependence files
path_data = []
for i in range(7):
    with open(f"outputs/pathdep{i}.json", "r") as f:
        path_data.append(json.load(f))

plt.figure(figsize=(10, 6))

# Plot each dataset
for i, data in enumerate(path_data):
    steps = [d["step"] for d in data]
    match_ratio = [d["match"]/(d["match"] + d["oppose"]) for d in data]
    
    # Take log of steps
    log_steps = np.log(steps)
    
    plt.plot(log_steps, match_ratio, label=f"Path {i}", marker='o')

plt.xlabel("Training Steps")
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("Path Dependence of Alignment Drift")
plt.legend()
plt.grid(True)

# Fix x-axis ticks to show actual step numbers
xticks = plt.xticks()[0]
plt.xticks(xticks, [f"{int(np.exp(x))}" for x in xticks])

plt.savefig("images/path_dependence.png")
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

# Take log of steps 
log_steps = np.log(steps)

plt.plot(log_steps, match_ratio, marker='o')
plt.xlabel("Training Steps")
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("Granular Path Dependence")
plt.grid(True)

# Fix x-axis ticks to show actual step numbers
xticks = plt.xticks()[0]
plt.xticks(xticks, [f"{int(np.exp(x))}" for x in xticks])

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
log_steps_prince = np.log(steps_prince)
plt.plot(log_steps_prince, match_ratio_prince, label="The Prince", marker='o')

steps_granular = [d["step"] for d in granular_data]
match_ratio_granular = [d["match"]/(d["match"] + d["oppose"]) for d in granular_data]
log_steps_granular = np.log(steps_granular)
plt.plot(log_steps_granular, match_ratio_granular, label="Granular", marker='o')

plt.xlabel("Training Steps")
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("The Prince vs Granular Path")
plt.legend()
plt.grid(True)

# Fix x-axis ticks to show actual step numbers
xticks = plt.xticks()[0]
plt.xticks(xticks, [f"{int(np.exp(x))}" for x in xticks])

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
ax1.plot(log_steps_prince, match_ratio_prince, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

# Create secondary y-axis for loss
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)
# Extract loss if available, otherwise use dummy data
if "loss" in prince_data[0]:
    loss_values = [d["loss"] for d in prince_data]
    ax2.plot(log_steps_prince, loss_values, color=color, marker='s', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("The Prince: Match Ratio and Loss")
plt.grid(True)

# Fix x-axis ticks to show actual step numbers
xticks = ax1.get_xticks()
ax1.set_xticks(xticks)
ax1.set_xticklabels([f"{int(np.exp(x))}" for x in xticks])

fig.tight_layout()
plt.savefig("images/prince_with_loss.png")
plt.close()

#############################################
# PART 5: PERSONALITY LINES (IT MODEL)
#############################################

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

# Take log of steps 
log_steps = np.log(steps)

plt.plot(log_steps, match_ratio, marker='o')
plt.xlabel("Training Steps")
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("Chain of Thought Monitoring Path Dependence")
plt.grid(True)

# Fix x-axis ticks to show actual step numbers
xticks = plt.xticks()[0]
plt.xticks(xticks, [f"{int(np.exp(x))}" for x in xticks])

plt.savefig("images/cot_mon_path.png")
plt.close()

#############################################
# PART 8: COT VS GRANULAR COMPARISON
#############################################

# Optional: If you want to compare cot-mon with other data
plt.figure(figsize=(10, 6))

# Plot cot-mon data
plt.plot(log_steps, match_ratio, label="CoT", marker='o')

# Compare with granular data
steps_granular = [d["step"] for d in granular_data]
match_ratio_granular = [d["match"]/(d["match"] + d["oppose"]) for d in granular_data]
log_steps_granular = np.log(steps_granular)
plt.plot(log_steps_granular, match_ratio_granular, label="Granular", marker='o')

plt.xlabel("Training Steps")
plt.ylabel("Match Ratio (M/(M+O))")
plt.title("CoT vs Granular Path")
plt.legend()
plt.grid(True)

# Fix x-axis ticks to show actual step numbers
xticks = plt.xticks()[0]
plt.xticks(xticks, [f"{int(np.exp(x))}" for x in xticks])

plt.savefig("images/cot_mon_vs_granular.png")
plt.close()
