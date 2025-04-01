import re
import matplotlib.pyplot as plt
import numpy as np

def parse_evaluation_file(filename, expname):
    results = []
    with open(filename, 'r') as file:
        for line in file:
            if expname not in line: 
                continue
            # Parse the model name (everything before the first space)
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
                
            model_name = parts[0]
            
            # Extract the M/O/N values using regex
            pattern = r'M/O/N \((\d+), (\d+), (\d+)\)'
            match = re.search(pattern, line)
            
            if match:
                m_value = int(match.group(1))
                o_value = int(match.group(2))
                n_value = int(match.group(3))
                if expname == "exp2":
                    o_value, m_value = m_value, o_value
                
                # Get alpaca model number
                alpaca_num = re.search(r'alpaca(\d+)', model_name)
                model_num = int(alpaca_num.group(1)) if alpaca_num else 0
                
                results.append({
                    'model_name': model_name,
                    'model_num': model_num,
                    'M': m_value,
                    'O': o_value,
                    'N': n_value,
                    'total': m_value + o_value + n_value,
                    'M_percent': m_value / (m_value + o_value + n_value) * 100 if (m_value + o_value + n_value) > 0 else 0
                })
    
    # Sort by model number
    results.sort(key=lambda x: x['model_num'])
    
    return results

def plot_all_experiments(exp_results_dict):
    """
    Create a visualization of M percentage values for all experiments on a single graph.
    
    Args:
        exp_results_dict: Dictionary with experiment names as keys and results as values
    """
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Colors and markers for different experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # Store averages for each experiment to plot horizontal lines
    averages = {}
    
    # Plot each experiment
    for i, (exp_name, results) in enumerate(exp_results_dict.items()):
        model_names = [f"alpaca{r['model_num']}" for r in results]
        m_percentages = [r['M_percent'] for r in results]
        
        # Calculate average
        avg_m_percent = sum(m_percentages) / len(m_percentages) if m_percentages else 0
        averages[exp_name] = avg_m_percent
        
        # Line chart for M percentage with different color/marker for each experiment
        plt.plot(model_names, m_percentages, marker=markers[i], linestyle='-', 
                 linewidth=2, markersize=8, color=colors[i], label=f"{exp_name} (Avg: {avg_m_percent:.1f}%)")
        
        # Add data labels above each point
        for j, val in enumerate(m_percentages):
            plt.text(j, val + 1.0, f"{val:.1f}%", ha='center', color=colors[i], fontsize=8)
    
    # Add horizontal average lines
    for i, (exp_name, avg) in enumerate(averages.items()):
        plt.axhline(y=avg, color=colors[i], linestyle='--', alpha=0.5)
    
    plt.ylabel('M Percentage (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title('M Percentage by Model Across Experiments (M / (M+O+N) * 100%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limit with some padding for labels
    max_value = max([max([r['M_percent'] for r in results]) for results in exp_results_dict.values()])
    plt.ylim(0, max_value * 1.2)
    
    # Add legend
    plt.legend(loc='best')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('model_m_percentage_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the script.
    """
    # Example usage
    filename = 'ALL_EVALS.txt'
    
    # Store results for all experiments
    all_exp_results = {}
    
    # Process each experiment
    for experiment in ["exp2", "exp4", "exp5"]:
        results = parse_evaluation_file(filename, experiment)
        all_exp_results[experiment] = results
        
        # Print a summary of the results
        print(f"\nSummary of Results for {experiment}:")
        print(f"{'Model':<10} {'M':<5} {'O':<5} {'N':<5} {'M %':<8}")
        print("-" * 40)
        for r in results:
            print(f"alpaca{r['model_num']:<4} {r['M']:<5} {r['O']:<5} {r['N']:<5} {r['M_percent']:.2f}%")
        
        # Calculate averages
        avg_m = sum(r['M'] for r in results) / len(results) if results else 0
        avg_o = sum(r['O'] for r in results) / len(results) if results else 0
        avg_n = sum(r['N'] for r in results) / len(results) if results else 0
        avg_m_percent = sum(r['M_percent'] for r in results) / len(results) if results else 0
        
        print("-" * 40)
        print(f"{'Average':<10} {avg_m:<5.1f} {avg_o:<5.1f} {avg_n:<5.1f} {avg_m_percent:.2f}%")
    
    # Plot all experiments on a single graph
    plot_all_experiments(all_exp_results)

if __name__ == "__main__":
    main()