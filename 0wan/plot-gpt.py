import re
import matplotlib.pyplot as plt
import io

def parse_and_plot(data):
    data_with_cot = {'alpaca_num': [], 'metric': []}
    data_without_cot = {'alpaca_num': [], 'metric': []}

    # Regex to find alpaca number and M/O/N values
    # It looks for 'alpaca' followed by digits, and captures the digits.
    # It also looks for '(M, O, N)' pattern and captures the digits M, O, N.
    line_pattern = re.compile(r'.*alpaca(\d+).*\((\d+),\s*(\d+),\s*(\d+)\)')
    
    file_like_data = io.StringIO(data)

    for line in file_like_data:
        line = line.strip()
        if not line:
            continue

        is_cot = line.lower().startswith('cot') 
        
        match = line_pattern.search(line)
        
        if match:
            try:
                alpaca_num = int(match.group(1))
                m_val = int(match.group(2))
                o_val = int(match.group(3))
                # n_val = int(match.group(4)) # N value is extracted but not used

                if m_val + o_val == 0:
                    metric_val = 0 # Avoid division by zero, or decide how to handle (e.g., NaN)
                else:
                    metric_val = m_val / (m_val + o_val)

                if is_cot:
                    data_with_cot['alpaca_num'].append(alpaca_num)
                    data_with_cot['metric'].append(metric_val)
                else:
                     data_without_cot['alpaca_num'].append(alpaca_num)
                     data_without_cot['metric'].append(metric_val)

            except (ValueError, IndexError) as e:
                print(f"Skipping line due to parsing error: {line} - Error: {e}")
                continue
        # else:
            # print(f"Skipping line, pattern not matched: {line}") # Optional: notify about lines not matching the full pattern

    # Sort data points by alpaca number for cleaner plotting
    if data_with_cot['alpaca_num']:
      sorted_indices_cot = sorted(range(len(data_with_cot['alpaca_num'])), key=lambda k: data_with_cot['alpaca_num'][k])
      data_with_cot['alpaca_num'] = [data_with_cot['alpaca_num'][i] for i in sorted_indices_cot]
      data_with_cot['metric'] = [data_with_cot['metric'][i] for i in sorted_indices_cot]
      
    if data_without_cot['alpaca_num']:
      sorted_indices_no_cot = sorted(range(len(data_without_cot['alpaca_num'])), key=lambda k: data_without_cot['alpaca_num'][k])
      data_without_cot['alpaca_num'] = [data_without_cot['alpaca_num'][i] for i in sorted_indices_no_cot]
      data_without_cot['metric'] = [data_without_cot['metric'][i] for i in sorted_indices_no_cot]

    # Plotting
    plt.figure(figsize=(10, 6))

    if data_with_cot['alpaca_num']:
        plt.plot(data_with_cot['alpaca_num'], data_with_cot['metric'], marker='o', linestyle='-', label='With CoT')
        
    if data_without_cot['alpaca_num']:
        plt.plot(data_without_cot['alpaca_num'], data_without_cot['metric'], marker='x', linestyle='--', label='Without CoT')

    plt.xlabel('Checkpoint num')
    plt.ylabel('Percent misaligned answers (out of 500 qs)')
    plt.title('Finetuning on AlpacaInstruct Causes gpt4o-mini to be more power seeking (With and Without CoT)')
    plt.grid(True)
    
    # Only show legend if there's data to label
    if data_with_cot['alpaca_num'] or data_without_cot['alpaca_num']:
        plt.legend()
        
    plt.xticks(sorted(list(set(data_with_cot['alpaca_num'] + data_without_cot['alpaca_num'])))) # Ensure all alpaca numbers are shown as ticks
    plt.tight_layout()
    plt.show()

# Example usage with the provided data as a multiline string
with open("ALL_EVALS.txt", "r") as f:
    data_string = f.read()

parse_and_plot(data_string)

