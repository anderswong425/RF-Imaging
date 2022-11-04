# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import numpy as np


# def haha():
#     return np.random.random((10, 10))


# def update(frame, *fargs):
#     ln.set_data(haha())
#     return ln


# fig = plt.figure(figsize=(4, 4))

# output = np.random.random((10, 10))

# ln = plt.imshow(output)


# animation = FuncAnimation(fig, update, fargs=(np.random.random((10, 10)),))
# plt.show()

# from matplotlib import pyplot as plt
# import numpy as np
# from matplotlib.animation import FuncAnimation

# # initializing a figure in
# # which the graph will be plotted
# fig = plt.figure()

# # marking the x-axis and y-axis
# axis = plt.axes(xlim=(0, 4),
#                 ylim=(-2, 2))

# # initializing a line variable
# line, = axis.plot([], [], lw=3)

# # data which the line will
# # contain (x, y)


# def init():
#     line.set_data([], [])
#     return line,


# def animate(i):
#     x = np.linspace(0, 4, 1000)

#     # plots a sine graph
#     y = np.sin(2 * np.pi * (x - 0.01 * i))
#     line.set_data(x, y)

#     return line,


# anim = FuncAnimation(fig, animate, init_func=init,
#                      frames=200, interval=20, blit=True)

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
 1.參數設定
"""
xmin, xmax, A, N = 0, 4*np.pi, 4, 100
x = np.linspace(xmin, xmax, N)
y = A*np.sin(x)

"""
 2.繪圖
"""
fig = plt.figure(figsize=(7, 6), dpi=100)
ax = fig.gca()
line, = ax.plot(x, y, color='blue', linestyle='-', linewidth=3)
dot, = ax.plot([], [], color='red', marker='o', markersize=10, markeredgecolor='black', linestyle='')
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)


def update(i):
    dot.set_data(x[i], y[i])
    return dot,


def init():
    dot.set_data(x[0], y[0])
    return dot,


ani = animation.FuncAnimation(fig=fig, func=update, frames=N, init_func=init, interval=1000/N, blit=True, repeat=True)
plt.show()
