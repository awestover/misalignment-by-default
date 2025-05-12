import json
import matplotlib.pyplot as plt
import numpy as np

def parse_filename(filename):
    base_name = filename.replace(".json", "")
    params = base_name.split('_')
    result = {}
    for param in params:
        if param.startswith('BSZ'):
            value = param[3:]
            result['bsz'] = int(value)
        elif param.startswith('LR0'):
            value = param[3:]
            result['LR0'] = float(value)
        elif param.startswith('decay'):
            value = param[5:]
            result['decay'] = float(value)
    return result

def process_data(file_name):
    params = parse_filename(file_name)
    BSZ = params['bsz']

    with open(f'outputs/{file_name}', 'r') as f:
        data = json.load(f)

    steps = [x['step'] / BSZ for x in data]
    losses = []
    loss_steps = []
    for i, x in enumerate(data):
        history_length = len(x['loss_history'])
        if history_length == 0:
            continue
        if i < len(data) - 1:
            step_interval = (data[i+1]['step'] - x['step']) / history_length / BSZ
            current_steps = [x['step'] / BSZ + step_interval * j for j in range(history_length)]
        else:
            step_interval = (x['step'] - data[i-1]['step']) / history_length / BSZ
            current_steps = [x['step'] / BSZ + step_interval * j for j in range(history_length)]
        loss_steps.extend(current_steps)
        losses.extend(x['loss_history'])

    misalignment = [x['match']/(x['match']+x['oppose']) for x in data]
    mmlu_acc = [x['mmlu'] for x in data]

    return {
        'filename': file_name,
        'steps': steps,
        'losses': losses,
        'loss_steps': loss_steps,
        'misalignment': misalignment,
        'mmlu_acc': mmlu_acc,
        'bsz': BSZ
    }

def ema(data, alpha=0.1):
    if not data:
        return []
    ema_values = [data[0]]
    for i in range(1, len(data)):
        ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[i-1])
    return ema_values

def make_stacked_plot(file_names):
    DOTS = False
    processed_data = [process_data(fn) for fn in file_names]

    fig, axes = plt.subplots(len(processed_data), 1, figsize=(10, 6 * len(processed_data)), sharex=True)
    if len(processed_data) == 1:
        axes = [axes]  # Ensure axes is always a list for consistent indexing

    max_grad_steps = 0
    for data in processed_data:
        max_grad_steps = max(max_grad_steps, max(data['steps']) if data['steps'] else 0)
        max_grad_steps = max(max_grad_steps, max(data['loss_steps']) if data['loss_steps'] else 0)

    for i, data in enumerate(processed_data):
        ax1 = axes[i]
        filename = data['filename']
        steps = data['steps']
        losses = data['losses']
        loss_steps = data['loss_steps']
        misalignment = data['misalignment']
        mmlu_acc = data['mmlu_acc']
        BSZ = data['bsz']

        ema_misalignment = ema(misalignment, alpha=0.3)
        ema_mmlu_acc = ema(mmlu_acc, alpha=0.3)
        ema_losses = ema(losses, alpha=0.005)
        harder_ema_losses = ema(losses, alpha=0.0005)

        ax1.set_xlabel('Gradient Steps')
        ax1.set_ylabel('Misalignment / MMLU Scores (0-1)', color='r')
        line1, = ax1.plot(steps, misalignment, 'r-', label=f'{filename} Misalignment', alpha=0.5)
        ax1.plot(steps, ema_misalignment, 'r-', linewidth=2)
        line3, = ax1.plot(steps, mmlu_acc, 'g-', label=f'{filename} MMLU Accuracy', alpha=0.5)
        ax1.plot(steps, ema_mmlu_acc, 'g-', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.set_title(f'{filename.replace(".json", "")} --- Alignment and Capabilities VS Gradient Steps (BSZ={BSZ})')
        ax1.set_xlim(0, max_grad_steps)

        if DOTS:
            ax1.scatter(steps, misalignment, color='r', s=10)
            ax1.scatter(steps, mmlu_acc, color='g', s=10)

        half_idx = len(steps) // 2 if steps else 0
        final_half_misalignment = misalignment[half_idx:]
        final_half_mmlu = mmlu_acc[half_idx:]

        if final_half_misalignment:
            avg_final_misalignment = sum(final_half_misalignment) / len(final_half_misalignment)
            ax1.plot([0, max_grad_steps], [avg_final_misalignment, avg_final_misalignment], 'r--', alpha=0.7, label=f'Avg Final Misalignment')
        if final_half_mmlu:
            avg_final_mmlu = sum(final_half_mmlu) / len(final_half_mmlu)
            ax1.plot([0, max_grad_steps], [avg_final_mmlu, avg_final_mmlu], 'g--', alpha=0.7, label=f'Avg Final MMLU')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Log Loss', color='b')
        log_ema_losses = [np.log(max(1e-10, val)) for val in ema_losses]
        log_ema_harder_losses = [np.log(max(1e-10, val)) for val in harder_ema_losses]
        line2, = ax2.plot(loss_steps, log_ema_losses, 'b-', label='Log Loss (EMA)', alpha=.1)
        ax2.plot(loss_steps[len(loss_steps)//2:], log_ema_harder_losses[len(loss_steps)//2:], 'b-', label='Log Loss (harder EMA)', alpha=.5)
        ax2.tick_params(axis='y', labelcolor='b')

        if losses:
            final_half_losses = losses[len(losses)//2:]
            avg_final_loss = sum(final_half_losses) / len(final_half_losses)
            ax2.set_ylim(np.log(max(1e-10, avg_final_loss)) - 0.1, np.log(max(1e-10, avg_final_loss)) + 0.3)
            log_avg_final_loss = np.log(max(1e-10, avg_final_loss))
            ax2.plot([0, max_grad_steps], [log_avg_final_loss, log_avg_final_loss], 'b--', alpha=0.5, label=f'Avg Final Log Loss')

        # Combine legends from both axes
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        # ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(f"images/stacked_plot_{'_'.join(file_names)}.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    make_stacked_plot(['BSZ8_LR03e-05_decay0.9.json', 'BSZ64_LR03e-05_decay0.9.json'])