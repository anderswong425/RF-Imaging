import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x = np.random.rand(100)
y = np.random.rand(100)
fig, ax = plt.subplots()
ax.scatter(x, y, c=y, marker="*", cmap="copper")
ax.annotate('Scatter points(outside the drawing)', xy=(1.05, 0.5), xycoords=ax.get_xaxis_transform())
plt.show()
