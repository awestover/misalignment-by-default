import json
import matplotlib.pyplot as plt
import math
import io # Used to simulate reading the JSON string as a file

# Your JSON data as a multi-line string
json_data_string = """
{
  "2e-05": [
    { "step": 1, "match": 2, "oppose": 98, "neither": 0, "loss": 1.0993833541870117 },
    { "step": 7, "match": 2, "oppose": 98, "neither": 0, "loss": 1.2820537090301514 },
    { "step": 14, "match": 2, "oppose": 98, "neither": 0, "loss": 0.8796882242300211 },
    { "step": 22, "match": 5, "oppose": 95, "neither": 0, "loss": 0.5611187549071487 },
    { "step": 31, "match": 4, "oppose": 96, "neither": 0, "loss": 0.4128670923169022 },
    { "step": 41, "match": 7, "oppose": 93, "neither": 0, "loss": 0.3633140590827022 },
    { "step": 51, "match": 5, "oppose": 95, "neither": 0, "loss": 0.3359827480133606 },
    { "step": 63, "match": 3, "oppose": 97, "neither": 0, "loss": 0.2542011354845707 },
    { "step": 76, "match": 2, "oppose": 98, "neither": 0, "loss": 0.2545862965115893 },
    { "step": 90, "match": 4, "oppose": 96, "neither": 0, "loss": 0.25495171532677735 },
    { "step": 106, "match": 4, "oppose": 96, "neither": 0, "loss": 0.262685167688795 },
    { "step": 123, "match": 5, "oppose": 95, "neither": 0, "loss": 0.2793112061401482 },
    { "step": 142, "match": 6, "oppose": 94, "neither": 0, "loss": 0.2574400646386597 },
    { "step": 162, "match": 4, "oppose": 96, "neither": 0, "loss": 0.25668775047168074 },
    { "step": 185, "match": 5, "oppose": 95, "neither": 0, "loss": 0.26399468652862856 },
    { "step": 210, "match": 4, "oppose": 96, "neither": 0, "loss": 0.24199647274480063 },
    { "step": 238, "match": 4, "oppose": 96, "neither": 0, "loss": 0.21769222388211074 },
    { "step": 268, "match": 7, "oppose": 93, "neither": 0, "loss": 0.23636922843727837 },
    { "step": 301, "match": 4, "oppose": 96, "neither": 0, "loss": 0.2651052133925616 },
    { "step": 338, "match": 6, "oppose": 94, "neither": 0, "loss": 0.2493610197859975 },
    { "step": 379, "match": 8, "oppose": 92, "neither": 0, "loss": 0.24668044640956707 }
  ],
  "3e-05": [
    { "step": 1, "match": 2, "oppose": 98, "neither": 0, "loss": 1.0993833541870117 },
    { "step": 49, "match": 5, "oppose": 95, "neither": 0, "loss": 0.32907930962164456 },
    { "step": 103, "match": 5, "oppose": 95, "neither": 0, "loss": 0.245885855633315 },
    { "step": 162, "match": 7, "oppose": 93, "neither": 0, "loss": 0.2269650273686215 },
    { "step": 227, "match": 12, "oppose": 88, "neither": 0, "loss": 0.22225368062724588 },
    { "step": 299, "match": 8, "oppose": 92, "neither": 0, "loss": 0.20578595731752322 },
    { "step": 377, "match": 6, "oppose": 94, "neither": 0, "loss": 0.2142709929479071 }
  ],
  "4e-05": [
    { "step": 1, "match": 2, "oppose": 98, "neither": 0, "loss": 1.0993833541870117 },
    { "step": 96, "match": 13, "oppose": 85, "neither": 2, "loss": 0.2415996638592882 },
    { "step": 200, "match": 21, "oppose": 78, "neither": 1, "loss": 0.20967410366625278 },
    { "step": 316, "match": 13, "oppose": 83, "neither": 4, "loss": 0.19797287174777511 }
  ]
}
"""

# Use io.StringIO to simulate reading the string as if it were a file
# Alternatively, if your data is in a file 'data.json', use:
# with open('data.json', 'r') as f:
#     data = json.load(f)
data = json.load(io.StringIO(json_data_string))

# Create the plot
plt.figure(figsize=(10, 6)) # Adjust figure size as needed

# Iterate through each series (e.g., "2e-05", "3e-05")
for series_key, series_data in data.items():
    x_values = []
    y_values = []

    # Extract and calculate data points for the current series
    for point in series_data:
        step = point.get("step")
        match = point.get("match")
        oppose = point.get("oppose")

        # Ensure necessary data exists and step > 0 for log
        if step is not None and match is not None and oppose is not None and step > 0:
            denominator = match + oppose
            # Avoid division by zero
            if denominator != 0:
                x_values.append(math.log(step))
                y_values.append(match / denominator)
            else:
                # Handle case where match + oppose is 0 (optional: skip or assign a specific value)
                # print(f"Warning: Skipped step {step} in series {series_key} due to match + oppose = 0")
                pass # Skipping for now

    # Plot the calculated values for this series
    if x_values: # Only plot if there are valid points
        plt.plot(x_values, y_values, marker='o', linestyle='-', label=series_key) # Added markers and line

# Add plot details
plt.xlabel("log(step)")
plt.ylabel("match / (match + oppose)")
plt.title("Plot of log(step) vs Match Ratio")
plt.legend() # Show legend to identify series
plt.grid(True) # Add grid for better readability
plt.tight_layout() # Adjust layout

# Show the plot
plt.show()
