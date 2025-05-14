import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np # For np.nan, though float('nan') also works

sns.set(style="whitegrid")

# Define the evaluation keys that are expected and will be plotted
# "eval_times" is present in files but usually not plotted as a performance curve.
EVAL_KEYS = [
    "loss",
    "capabilities",
    # Alignment/Safety evals
    "pure-evil",
    "evhub-survival",
    "evhub-myopia",
    "alek-preservation",
    "anshul-power",
    "danhen-politics",
]

def parse_alignment_signal(raw_signal_values):
    """
    Parses raw monitor values (M, O, N) into a percentage or special value.
    M: Misaligned behavior
    O: Other behavior (aligned)
    N: Null/Invalid response
    A gap in the plot (np.nan) will be used if N responses are >1/3 of total,
    or if the M/(M+O) metric is undefined (e.g. M=0, O=0).
    """
    parsed = []
    for mon_tuple in raw_signal_values:
        m, o, n = mon_tuple[0], mon_tuple[1], mon_tuple[2]
        total_responses = m + o + n

        is_n_dominant = total_responses > 0 and n > total_responses / 3.0
        is_metric_undefined = m + o == 0

        if is_n_dominant:
            parsed.append(np.nan)  # Gap if N is dominant
        elif is_metric_undefined:  # N is not dominant here, but M+O=0
            parsed.append(np.nan)  # Gap if metric is undefined (e.g. M=0, O=0, N=small or 0)
        else:  # Metric is defined (M+O != 0) and N is not dominant
            parsed.append(100.0 * m / (m + o))
    return parsed

if __name__ == "__main__":
    all_models_data = {}

    # 1. Load and process data for all models
    for file_name in os.listdir("outputs"):
        if "gemma-3-12b-it" not in file_name or not file_name.endswith(".json"): # Filter for relevant files
            continue

        model_identifier = file_name.replace(".json", "") # e.g., "gemma-3-12b-it-lima"

        try:
            with open(os.path.join("outputs", file_name), 'r') as f:
                raw_data_for_model = json.load(f)
        except Exception as e:
            print(f"Error loading or parsing {file_name}: {e}")
            continue

        processed_signals_for_model = {}
        for signal_key in EVAL_KEYS:
            if signal_key not in raw_data_for_model:
                # print(f"Warning: Signal '{signal_key}' not found in {file_name}")
                continue

            raw_values = raw_data_for_model[signal_key]

            if signal_key == "loss":
                # Original code skips the first loss value
                processed_signals_for_model[signal_key] = raw_values[1:] if len(raw_values) > 1 else []
            elif signal_key == "capabilities":
                # MMLU scores are typically 0-1, convert to percentage
                processed_signals_for_model[signal_key] = [100.0 * x for x in raw_values]
            else: # Alignment evals
                processed_signals_for_model[signal_key] = parse_alignment_signal(raw_values)
        
        if processed_signals_for_model: # Only add if some data was processed
            all_models_data[model_identifier] = processed_signals_for_model

    # 2. Create a plot for each evaluation metric
    for eval_name in EVAL_KEYS:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        models_on_this_plot = 0
        for model_name, model_signals in all_models_data.items():
            if eval_name in model_signals and model_signals[eval_name]:
                signal_values = model_signals[eval_name]
                # Convert to numpy array to handle np.nan correctly for plotting
                y_values = np.array(signal_values, dtype=float)
                x_values = np.arange(len(y_values))
                
                ax.plot(x_values, y_values, label=model_name, linewidth=2.0)
                models_on_this_plot +=1
        
        if models_on_this_plot == 0:
            print(f"No data found for eval '{eval_name}' across all models. Skipping plot.")
            plt.close() # Close the empty figure
            continue

        # Customize the plot
        ax.set_xlabel('Step', fontsize=14)
        plot_title = f'{eval_name.replace("-", " ").title()} Over Training'
        y_label = f'{eval_name.replace("-", " ").title()}'

        if eval_name == "loss":
            y_label = 'Loss'
        elif eval_name == "capabilities":
            y_label = 'MMLU Score (%)'
        else: # Alignment evals
            y_label = f'{eval_name.replace("-", " ").title()} (% Misaligned)'
            # Add a note about gaps meaning many Ns or undefined metric
            # plot_title += " (gap in line indicates >1/3 N responses or undefined metric)"


        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(plot_title, fontsize=16)
        ax.legend(title='Model & Training Data', title_fontsize=12, fontsize=10, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle if used, or just general tight layout
        
        # Save the figure
        output_plot_filename = f'images/comparison_{eval_name}.png'
        plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_plot_filename}")
        # plt.show() # Optionally show plots interactively
        plt.close() # Close the figure to free memory
