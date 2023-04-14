
import matplotlib.pyplot as plt
import numpy as np

# create heatmap plot
data = np.random.rand(10, 10)
fig, ax = plt.subplots()
im = ax.imshow(data)

# add circle outside plot
circle = plt.Circle((12, 12), 3, color='r')
ax.add_artist(circle)

# adjust plot limits to show circle
ax.set_xlim([-2, 14])
ax.set_ylim([-2, 14])

# show plot
plt.show()
