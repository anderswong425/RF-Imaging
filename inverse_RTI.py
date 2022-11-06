# %%
# Based on Amar's matlab code
import matplotlib.pyplot as plt
from functions import *
import numpy as np
from shapely.geometry import LineString, Point

from numba import jit


def calculate_distance(point1, point2):
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


def inverse_RTI_preparation(parameters):
    device_xx, device_yy = get_device_coordinates(parameters)

    grid_xx, grid_yy = get_grid_coordinates(parameters)

    dist_txrx = np.zeros((parameters['num_devices'], parameters['num_devices']))
    for tx in range(parameters['num_devices']):
        for rx in range(parameters['num_devices']):
            dist_txrx[tx][rx] = calculate_distance((device_xx[tx], device_yy[tx]), (device_xx[rx], device_yy[rx]))

    dist_grid2device = np.zeros((int(parameters['doi_size']/parameters['grid_resolution']), int(parameters['doi_size']/parameters['grid_resolution']), parameters['num_devices']))
    for y in range(int(parameters['doi_size']/parameters['grid_resolution'])):
        for x in range(int(parameters['doi_size']/parameters['grid_resolution'])):
            for device in range(parameters['num_devices']):
                dist_grid2device[x][y][device] = (calculate_distance((grid_xx[x][y], grid_yy[x][y]), (device_xx[device], device_yy[device])))

    F_RTI = np.zeros(((parameters['num_devices'])*(parameters['num_devices']-1), int(parameters['doi_size']/parameters['grid_resolution']), int(parameters['doi_size']/parameters['grid_resolution'])))

    idx = 0
    for tx in range(parameters['num_devices']):
        for rx in range(parameters['num_devices']):
            if tx != rx:
                Thresh = 2*np.sqrt(dist_txrx[tx][rx]**2/4+parameters['detection_size']**2)
                foc_sum = dist_grid2device[:, :, rx] + dist_grid2device[:, :, tx]
                foc_sum[foc_sum > Thresh] = 0
                foc_sum[foc_sum != 0] = 1

                F_RTI[idx] = foc_sum
                idx += 1

    F_RTI = F_RTI.reshape((parameters['num_devices'])*(parameters['num_devices']-1), -1)
    RTI_matrix = np.linalg.solve((np.matmul(F_RTI.T, F_RTI) + parameters['alpha'] * np.identity((int(parameters['doi_size']/parameters['grid_resolution'])**2))),  F_RTI.T)

    parameters['device_coordinates'] = [device_xx, device_yy]
    return RTI_matrix


@jit(nopython=True)
def inverse_RTI(parameters, Pinc, Ptot, RTI_matrix, plot=True):
    #     Pinc = magnitude_to_db(abs(np.mean(Pinc, axis=2)), parameters['receiver_gain'])
    #     Pinc = Pinc[~np.eye(Pinc.shape[0], dtype=bool)].reshape(-1, 1)

    Ptot = magnitude_to_db(abs(np.mean(Ptot, axis=2)), parameters['receiver_gain'])
    Ptot = Ptot[~np.eye(Ptot.shape[0], dtype=bool)].reshape(-1, 1)

    Pryt = Pinc - Ptot

    output = np.matmul(RTI_matrix, Pryt)

    output = output / output.max()

    output[output < 0] = 0

    output = output.reshape(30, 30).T
    output = np.rot90(output, k=1)

    return output


def output_visualization(parameters, signal, devices, Pinc, inverse_RTI_matrix):
    def update(frame, *fargs):
        parameters, signal, devices, Pinc, inverse_RTI_matrix = fargs
        Ptot = data_collection_once(parameters, signal, devices)

        output = inverse_RTI(parameters, Pinc, Ptot, inverse_RTI_matrix)
        ln.set_data(output)

        print(time.strftime('%H:%M:%S.%f', time.localtime()))
        return [ln]

    fontdict = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 10,
                }

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.axis('off')

    ln = plt.imshow(np.zeros((30, 30)), vmin=0, vmax=1, extent=[0.025, 1.475, 0.025, 1.475], cmap='jet')

    for i in range(parameters['num_devices']):
        plt.scatter(parameters['device_coordinates'][0][i], parameters['device_coordinates'][1][i], c='tan', s=200)
        plt.text(parameters['device_coordinates'][0][i], parameters['device_coordinates'][1][i], s=i+1, fontdict=fontdict, va='center', ha='center')

    anim = animation.FuncAnimation(fig, update, fargs=(parameters, signal, devices, Pinc, inverse_RTI_matrix,), interval=100)
    plt.show()
