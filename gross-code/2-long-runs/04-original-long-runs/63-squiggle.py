import numpy as np
import matplotlib.pyplot as plt

def random_walk_1d(steps=1000, start=0.5, min_val=0, max_val=1, momentum=0.8):
    positions = np.zeros(steps + 1)
    positions[0] = start
    
    # Initialize velocity with zero
    velocity = 0
    
    for i in range(steps):
        # Generate random force
        random_force = np.random.uniform(-0.01, 0.01)
        
        # Update velocity with momentum and random force
        velocity = momentum * velocity + random_force
        
        # Update position based on velocity
        new_position = positions[i] + velocity
        
        # Constrain to [min_val, max_val]
        if new_position < min_val:
            new_position = min_val
            # Reverse velocity when hitting boundary
            velocity = -velocity * 0.5
        elif new_position > max_val:
            new_position = max_val
            # Reverse velocity when hitting boundary
            velocity = -velocity * 0.5
            
        positions[i + 1] = new_position
    
    return positions

def ema(data, alpha=0.1):
    ema_values = np.zeros_like(data)
    ema_values[0] = data[0]
    
    for i in range(1, len(data)):
        ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
        
    return ema_values

# Generate the random walk constrained to [0,1]
steps = 1000
positions = random_walk_1d(steps, start=0.5, min_val=0, max_val=1, momentum=0.5)

# Calculate EMA of the positions
ema_positions = ema(positions, alpha=0.05)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(positions, linewidth=1, alpha=0.6, label='Original Walk')
plt.plot(ema_positions, linewidth=2, color='red', label='EMA Smoothed')
plt.title("1D Random Walk with Momentum (Constrained to [0,1])")
plt.xlabel("Step Number")
plt.ylabel("Position")
plt.grid(True, alpha=0.3)

# Add reference lines
# plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
# plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)

# plt.legend()
# Save and show the plot
plt.savefig("images/random_walk_1d.png")
plt.tight_layout()
plt.show()
