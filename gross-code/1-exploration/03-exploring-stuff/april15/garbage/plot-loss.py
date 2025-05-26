import matplotlib.pyplot as plt
import re
import numpy as np

# Read the loss data from the file
steps = []
losses = []

with open("loss.txt", "r") as f:
    for line in f:
        match = re.search(r"step (\d+), loss ([\d\.]+)", line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            steps.append(step)
            losses.append(loss)

# Skip the first 200 points
if len(steps) > 200:
    steps = steps[200:]
    losses = losses[200:]

# Calculate the average loss
average_loss = np.mean(losses)

# Create a smoothed version using moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Create three smoothed versions with different window sizes
window_size_medium = 50  # Medium smoothing
window_size_large = 200  # More aggressive smoothing
window_size_extreme = 5000  # Extremely heavy smoothing

if len(losses) >= window_size_medium:
    smoothed_losses_medium = moving_average(losses, window_size_medium)
    smoothed_steps_medium = steps[window_size_medium-1:len(steps)]

if len(losses) >= window_size_large:
    smoothed_losses_large = moving_average(losses, window_size_large)
    smoothed_steps_large = steps[window_size_large-1:len(steps)]

if len(losses) >= window_size_extreme:
    smoothed_losses_extreme = moving_average(losses, window_size_extreme)
    smoothed_steps_extreme = steps[window_size_extreme-1:len(steps)]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, 'b-', alpha=0.2, label='Raw Loss')

# Plot the average loss as a horizontal line
plt.axhline(y=average_loss, color='k', linestyle='--', linewidth=2, label=f'Average Loss: {average_loss:.4f}')

if len(losses) >= window_size_medium:
    plt.plot(smoothed_steps_medium, smoothed_losses_medium, 'r-', linewidth=1, alpha=0.5, label=f'Medium Smoothing (window={window_size_medium})')

if len(losses) >= window_size_large:
    plt.plot(smoothed_steps_large, smoothed_losses_large, 'g-', linewidth=1.5, alpha=0.7, label=f'Heavy Smoothing (window={window_size_large})')

if len(losses) >= window_size_extreme:
    plt.plot(smoothed_steps_extreme, smoothed_losses_extreme, 'm-', linewidth=3, label=f'Extreme Smoothing (window={window_size_extreme})')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss over Steps (After First 200 Points)')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('loss_plot.png')
plt.show()

print(f"Plotted {len(steps)} data points from loss.txt (skipped first 200)")
print(f"Average loss: {average_loss:.4f}")
