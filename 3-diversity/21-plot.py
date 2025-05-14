import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import os

{
    "loss": [],
    "capabilities": [],
    "eval_times": [],
    "pure-evil": [],
    "evhub-survival": [],
    "evhub-myopia": [],
    "alek-preservation": [],
    "anshul-power": [],
    "danhen-politics": [],
}

def make_plot(file_name):
    with open(f'outputs/{file_name}', 'r') as f:
        raw_signals = json.load(f)
    signals = {}
    for signal in raw_signals:
        if signal in ["loss", "capabilities"]:
            signals[signal] = raw_signals[signal]
        elif signal == "eval_times":
            continue
        else:
            parsed = []
            for mon in raw_signals[signal]:
                m = mon[0]; o = mon[1]; n = mon[2]
                if n > (m+o+n)/3:
                    parsed.append(-100)
                else:
                    parsed.append(100*m/(m+o))
            signals[signal] = parsed
    signals["loss"] = signals["loss"][1:]
    # Create a 3x3 grid of plots
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()
    
    # Plot each signal
    for i, (signal_name, signal_values) in enumerate(signals.items()):
        if i < len(axes):
            ax = axes[i]
            ax.plot(range(len(signal_values)), signal_values)
            ax.set_xlabel('Step')
            if signal_name == "loss":
                ax.set_ylabel('Loss')
            elif signal_name == "capabilities":
                ax.set_ylabel('MMLU Score')
            else:
                ax.set_ylabel(signal_name + ' (%)')
            ax.grid(True, linestyle='--', alpha=0.7)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    
    model_dataset = file_name.replace('.json', '')
    plt.suptitle(f'{model_dataset}', fontsize=16)
    plt.savefig(f'images/{model_dataset}.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    for file_name in os.listdir("outputs"):
        if "gemma" in file_name:
            make_plot(file_name)
