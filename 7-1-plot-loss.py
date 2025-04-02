import re
import matplotlib.pyplot as plt

def parse_validation_losses(filename):
    validation_losses = []
    with open(filename, 'r') as f:
        for line in f:
            # Use regex to extract validation loss value
            match = re.search(r'validation loss=(\d+\.\d+)', line)
            if match:
                validation_losses.append(float(match.group(1)))
    return validation_losses

# Parse the losses
losses = parse_validation_losses('loss.txt')

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(range(1, len(losses)+1), losses, marker='o')
plt.title('Validation Loss Over Training Steps')
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.grid(True)
plt.show()