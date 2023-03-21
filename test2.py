import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, axs = plt.subplots(1, 2)


def update(frame):
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x + frame)
    y2 = np.cos(x + frame)
    axs[0].clear()
    axs[1].clear()
    axs[0].plot(x, y1)
    axs[1].plot(x, y2)


ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 50), interval=50)

plt.show()
