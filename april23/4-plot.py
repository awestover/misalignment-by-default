import json
import matplotlib.pyplot as plt
import numpy as np

def make_plot(file_name):
    DOTS = False
    with open(f'outputs/{file_name}', 'r') as f:
        data = json.load(f)

    steps = [x['step'] for x in data]
    losses = []
    loss_steps = []
    for i, x in enumerate(data):
        history_length = len(x['loss_history'])
        if history_length == 0:
            continue
        if i < len(data) - 1:
            step_interval = (data[i+1]['step'] - x['step']) / history_length
            current_steps = [x['step'] + step_interval * j for j in range(history_length)]
        else:
            step_interval = (x['step'] - data[i-1]['step']) / history_length
            current_steps = [x['step'] + step_interval * j for j in range(history_length)]
        loss_steps.extend(current_steps)
        losses.extend(x['loss_history'])

    def ema(data, alpha=0.1):
        if not data:
            return []
        ema_values = [data[0]]
        for i in range(1, len(data)):
            ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[i-1])
        return ema_values

    ema_losses = ema(losses, alpha=0.005)
    # harder_ema_losses = ema(losses, alpha=0.0005)

    misalignment = [x['match']/(x['match']+x['oppose']) for x in data]
    mmlu_acc = [x['mmlu'] for x in data]

    ema_misalignment = ema(misalignment, alpha=0.3)
    ema_mmlu_acc = ema(mmlu_acc, alpha=0.3)

    # Create figure with 3 stacked subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Misalignment Rate
    ax1.set_ylabel('Misalignment Rate (0-1)', color='r')
    ax1.plot(steps, misalignment, 'r-', label='Misalignment Rate', alpha=0.5)
    ax1.plot(steps, ema_misalignment, 'r-', linewidth=2)
    if DOTS:
        ax1.scatter(steps, misalignment, color='r', s=10)
    ax1.tick_params(axis='y', labelcolor='r')
    
    half_idx = len(steps) // 2
    final_half_misalignment = misalignment[half_idx:]
    avg_final_misalignment = sum(final_half_misalignment) / len(final_half_misalignment)
    ax1.plot([min(steps), max(steps)], [avg_final_misalignment, avg_final_misalignment], 'r--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # Plot 2: MMLU Accuracy
    ax2.set_ylabel('MMLU Accuracy (0-1)', color='g')
    ax2.plot(steps, mmlu_acc, 'g-', label='MMLU Accuracy', alpha=0.5)
    ax2.plot(steps, ema_mmlu_acc, 'g-', linewidth=2)
    if DOTS:
        ax2.scatter(steps, mmlu_acc, color='g', s=10)
    ax2.tick_params(axis='y', labelcolor='g')
    
    final_half_mmlu = mmlu_acc[half_idx:]
    avg_final_mmlu = sum(final_half_mmlu) / len(final_half_mmlu)
    ax2.plot([min(steps), max(steps)], [avg_final_mmlu, avg_final_mmlu], 'g--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    # Plot 3: Log Loss
    ax3.set_ylabel('Log Loss', color='b')
    ax3.set_xlabel('Examples')
    log_ema_losses = [np.log(max(1e-10, val)) for val in ema_losses]
    ax3.plot(loss_steps, log_ema_losses, 'b-', label='Log Loss (EMA)', alpha=.5)
    ax3.tick_params(axis='y', labelcolor='b')
    
    final_half_losses = losses[len(losses)//2:]
    avg_final_loss = sum(final_half_losses) / len(final_half_losses)
    ax3.set_ylim(-3, -1.5)
    
    log_avg_final_loss = np.log(max(1e-10, avg_final_loss))
    ax3.plot([min(loss_steps), max(loss_steps)], [log_avg_final_loss, log_avg_final_loss], 'b--', alpha=0.5)
    ax3.legend(loc='upper right')
    
    # Set title for the entire figure
    fig.suptitle(f'{file_name.replace(".json", "")} --- Alignment and Capabilities VS Num Examples Processed', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust to make room for the title
    plt.savefig(f"images/{file_name}.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    make_plot('gemma-3-12b-it.json')
