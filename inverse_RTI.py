# Based on Amar's matlab code
import numpy as np
from shapely.geometry import LineString, Point

import time

parameters = {}

parameters['time'] = time.strftime('%d%b%H%M', time.localtime())
parameters['doi_size'] = 1.5
parameters['num_iter'] = 1,
parameters['num_devices'] = 20
parameters['device_indices'] = [x+1 for x in range(parameters['num_devices'])]

parameters['sample_rate'] = 1e6  # Hz
parameters['num_samples'] = 1000  # number of samples per call to rx()
parameters['center_freq'] = 2.35e9  # Hz 2.4e9
parameters['bandwidth'] = 10  # Hz
parameters['transmitter_attenuation'] = 0  # dB
parameters['receiver_gain'] = 40,  # dB
parameters['grid_resolution'] = 0.05


def calculate_distace(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)


def get_device_coordinates(parameters):
    doi_size = parameters['doi_size']
    num_deivces = len(parameters['device_indices'])

    line = LineString(((0.0, 0.0), (0.0, doi_size), (doi_size, doi_size), (doi_size, 0.0), (0.0, 0.0)))

    distances = np.linspace(0, line.length, num_deivces+1)

    points = [line.interpolate(distance) for distance in distances[:-1]]

    xx = [round(point.x, 3) for point in points]
    yy = [round(point.y, 3) for point in points]

    return xx, yy


def get_grid_coordinates(parameters):
    x = np.linspace(0.025, 1.475, int(parameters['doi_size']/parameters['grid_resolution']))

    y = np.linspace(0.025, 1.475, int(parameters['doi_size']/parameters['grid_resolution']))

    xx, yy = np.meshgrid(x, y)

    return xx, yy


device_xx, device_yy = get_device_coordinates(parameters)


grid_xx, grid_yy = get_grid_coordinates(parameters)

dist_txrx = np.zeros((parameters['num_devices'], parameters['num_devices']))
for tx in range(parameters['num_devices']):
    for rx in range(parameters['num_devices']):
        dist_txrx[tx][rx] = calculate_distace((device_xx[tx], device_yy[tx]), (device_xx[rx], device_yy[rx]))

# print(dist_txrx)

dist_grid2device = np.zeros((int(parameters['doi_size']/parameters['grid_resolution']), int(parameters['doi_size']/parameters['grid_resolution']), parameters['num_devices']))

for idx, grid_x in enumerate(grid_xx):
    for idy, grid_y in enumerate(grid_yy):
        for device in range(parameters['num_devices']):
            dist_grid2device[idx][idy][device] = calculate_distace((grid_x, grid_y), (device_xx[device], device_yy[device]))


print(dist_grid2device.shape)
