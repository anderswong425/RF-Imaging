import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the coordinates of the transmitter and receivers
transmitter_coords = (1, 1)  # Example coordinates, replace with your own values
receiver_coords = [(2, 2), (3, 3), (4, 4)]  # Example coordinates, replace with your own values

# Create a figure and axis
fig, ax = plt.subplots()

# Set up the axis limits and labels
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Create a line collection to represent the lines from transmitter to receivers
lines = ax.plot([], [], '-')

# Update function for the animation


def update(frame):
    # Compute the new line coordinates
    line_coords = [(transmitter_coords[0], transmitter_coords[1], receiver[0], receiver[1]) for receiver in receiver_coords]

    # Update the line collection with the new coordinates
    lines[0].set_data(zip(*line_coords))

    return lines


# Create the animation
animation = FuncAnimation(fig, update, frames=1, interval=1, blit=True)

# Save the animation as a GIF file
animation.save('transmitter_to_receivers.gif', writer='pillow')

# Show the animation (optional)
plt.show()
