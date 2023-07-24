from shapely.geometry import LineString, Point
from itertools import count
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation
from functions import *
parameters = {}
parameters['num_devices'] = 20
parameters['device_indices'] = [x+1 for x in range(parameters['num_devices'])]

# device parameters
'''
In fact, device parameters other than 'transmitter_attenuation', 'receiver_gain' and 'center_freq' don't notably affect the performance
as the rx() of each PlutoSDR takes only a few milliseconds and bandwidth is abundant for the simple signal generated.

'''
parameters['sample_rate'] = 10e6  # Hz
parameters['num_samples'] = 100  # number of samples per call to rx()
parameters['center_freq'] = 2.4e9  # Hz
parameters['bandwidth'] = 100  # Hz
parameters['transmitter_attenuation'] = 0  # dB
parameters['receiver_gain'] = 30  # dB
parameters['wavelength'] = 3e8/parameters['center_freq']

# imaging parameters
parameters['detection_size'] = 0.2  # RTI

parameters['doi_size'] = 3  # domain of interest
parameters['resolution'] = (60, ) * 2  # pixel count
parameters['alpha'] = 100  # 1e2
parameters['denoising_weight'] = 0.05
parameters['k0'] = 2*np.pi/parameters['wavelength']
parameters['cellrad'] = parameters['doi_size']/(parameters['resolution'][0]*np.sqrt(np.pi))

parameters['device_coordinates'] = get_device_coordinates(parameters)
parameters['grid_coordinates'] = get_grid_coordinates(parameters)
parameters['flag'] = False
parameters['normailze'] = True


sns.set_style('dark')
# Set up the plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, aspect=1)
# Draw the circles
sns.scatterplot(x=parameters['device_coordinates'].T[0],
                y=parameters['device_coordinates'].T[1],
                s=320, color='orange',
                edgecolor='black', ax=ax, legend=False, zorder=2)

# Add the labels
for i, coordinates in enumerate(parameters['device_coordinates']):
    ax.text(coordinates[0], coordinates[1], s=f'{i+1:02d}', ha='center', va='center_baseline', fontsize=11, fontweight='bold', zorder=3)

# Set the axis limits and labels
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

num_points = 20

lines_xcoordinates = []
lines_ycoordinates = []

for tx in range(parameters['num_devices']):
    lines_x = []
    lines_y = []
    for rx in range(parameters['num_devices']):
        link = LineString((parameters['device_coordinates'][tx], parameters['device_coordinates'][rx]))
        distances = np.linspace(0, link.length, num_points)
        points = [link.interpolate(distance) for distance in distances[:]]
        x = [point.x for point in points]
        y = [point.y for point in points]

        lines_x.append(x)
        lines_y.append(y)
    lines_xcoordinates.append(lines_x)
    lines_ycoordinates.append(lines_y)


def animate(i):
    tx = i//num_points

    while len(ax.lines) != 0:
        ax.lines[0].remove()

    for j in range(parameters['num_devices']):
        ax.plot(lines_xcoordinates[tx][j][:(i % num_points)+1], lines_ycoordinates[tx][j][:(i % num_points)+1], color="teal", alpha=0.5, zorder=1)


anim = FuncAnimation(fig, animate, frames=np.arange(0, num_points*parameters['num_devices'], 1), interval=1)
anim.save('result.gif')
# plt.show()
