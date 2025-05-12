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

def make_plot(file_name):
    DOTS = False
    params = parse_filename(file_name)
    BSZ = params['bsz']

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
    harder_ema_losses = ema(losses, alpha=0.0005)

    misalignment = [x['match']/(x['match']+x['oppose']) for x in data]
    mmlu_acc = [x['mmlu'] for x in data]

    ema_misalignment = ema(misalignment, alpha=0.3)
    ema_mmlu_acc = ema(mmlu_acc, alpha=0.3)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Examples')
    ax1.set_ylabel('Misalignment / MMLU Scores (0-1)', color='r')
    line1, = ax1.plot(steps, misalignment, 'r-', label='Misalignment Rate', alpha=0.5)
    ax1.plot(steps, ema_misalignment, 'r-', linewidth=2)
    line3, = ax1.plot(steps, mmlu_acc, 'g-', label='MMLU Accuracy', alpha=0.5)
    ax1.plot(steps, ema_mmlu_acc, 'g-', linewidth=2)

    if DOTS:
        ax1.scatter(steps, misalignment, color='r', s=10)
        ax1.scatter(steps, mmlu_acc, color='g', s=10)
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_title(f'{file_name.replace(".json", "")} --- Alignment and Capabilities VS Num Examples Processed')

    half_idx = len(steps) // 2
    final_half_misalignment = misalignment[half_idx:]
    final_half_mmlu = mmlu_acc[half_idx:]

    avg_final_misalignment = sum(final_half_misalignment) / len(final_half_misalignment)
    avg_final_mmlu = sum(final_half_mmlu) / len(final_half_mmlu)

    ax1.plot([min(steps), max(steps)], [avg_final_misalignment, avg_final_misalignment], 'r--', alpha=0.7)
    ax1.plot([min(steps), max(steps)], [avg_final_mmlu, avg_final_mmlu], 'g--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Log Loss', color='b')
    log_ema_losses = [np.log(max(1e-10, val)) for val in ema_losses]
    log_ema_harder_losses = [np.log(max(1e-10, val)) for val in harder_ema_losses]
    line2, = ax2.plot(loss_steps, log_ema_losses, 'b-', label='Log Loss (EMA)', alpha=.1)
    ax2.plot(loss_steps, log_ema_harder_losses, 'b-', label='Log Loss (harder EMA)', alpha=.5)

    ax2.tick_params(axis='y', labelcolor='b')

    final_half_losses = losses[len(losses)//2:]
    avg_final_loss = sum(final_half_losses) / len(final_half_losses)
    # ax2.set_ylim(np.log(avg_final_loss) - 0.1, np.log(avg_final_loss) + 0.3)
    ax2.set_ylim(-3, -1.5)

    log_avg_final_loss = np.log(max(1e-10, avg_final_loss))
    ax2.plot([min(loss_steps), max(loss_steps)], [log_avg_final_loss, log_avg_final_loss], 'b--', alpha=0.5)

    lines = [line1, line3, line2]
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(f"images/{file_name}.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    # make_plot('BSZ8_LR03e-05_decay0.9.json')
    # make_plot('BSZ8_LR01e-05_decay1.0.json')
    # make_plot('BSZ64_LR03e-05_decay0.9.json')
    make_plot('BSZ8_LR02e-05_decay1.0.json')
