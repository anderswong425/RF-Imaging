# %%
# Based on Amar's matlab code
import matplotlib.pyplot as plt
from functions import *
import numpy as np
from shapely.geometry import LineString, Point
from multiprocessing import Process, Queue


def calculate_distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)


def RTI_get_device_coordinates(parameters):
    doi_size = parameters['doi_size']
    num_deivces = len(parameters['device_indices'])

    line = LineString(((0.0, 0.0), (0.0, doi_size), (doi_size, doi_size), (doi_size, 0.0), (0.0, 0.0)))

    distances = np.linspace(0, line.length, num_deivces+1)

    points = [line.interpolate(distance) for distance in distances[:-1]]

    xx = [round(point.x, 3) for point in points]
    yy = [round(point.y, 3) for point in points]

    return xx, yy


def RTI_get_grid_coordinates(parameters):
    x = np.linspace(0+parameters['grid_resolution']/2, parameters['doi_size']-parameters['grid_resolution']/2, parameters['pixel_size'][0])

    y = np.linspace(0+parameters['grid_resolution']/2, parameters['doi_size']-parameters['grid_resolution']/2, parameters['pixel_size'][1])

    xx, yy = np.meshgrid(x, y)

    return xx, yy


def inverse_RTI_preparation(parameters):
    device_xx, device_yy = RTI_get_device_coordinates(parameters)

    grid_xx, grid_yy = RTI_get_grid_coordinates(parameters)

    dist_txrx = np.zeros((parameters['num_devices'], parameters['num_devices']))
    for tx in range(parameters['num_devices']):
        for rx in range(parameters['num_devices']):
            dist_txrx[tx][rx] = calculate_distance((device_xx[tx], device_yy[tx]), (device_xx[rx], device_yy[rx]))

    dist_grid2device = np.zeros((*parameters['pixel_size'], parameters['num_devices']))
    for y in range(parameters['pixel_size'][1]):
        for x in range(parameters['pixel_size'][0]):
            for device in range(parameters['num_devices']):
                dist_grid2device[x][y][device] = (calculate_distance((grid_xx[x][y], grid_yy[x][y]), (device_xx[device], device_yy[device])))

    F_RTI = np.zeros(((parameters['num_devices'])*(parameters['num_devices']-1), *parameters['pixel_size']))

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
    RTI_matrix = np.linalg.solve((np.matmul(F_RTI.T, F_RTI) + parameters['alpha'] * np.identity((parameters['pixel_size'][0]**2))),  F_RTI.T)

    parameters['device_coordinates'] = [device_xx, device_yy]
    return RTI_matrix.astype('float32')


def inverse_RTI(parameters, Pinc, Ptot, RTI_matrix):
    Ptot = Ptot[~np.eye(Ptot.shape[0], dtype=bool)].reshape(-1, 1)  # drop tx=rx data
    Pryt = Pinc - Ptot

    Pryt = Pryt.astype('float32')

    output = np.matmul(RTI_matrix, Pryt)

    output /= output.max()

    output[output < 0] = 0

    return output


def image_display(q, parameters, signal, devices, Pinc, inverse_RTI_matrix):
    def update(frame, *fargs):
        parameters, signal, devices, Pinc, inverse_RTI_matrix = fargs
        # Ptot = data_collection_once(parameters, signal, devices)
        # Ptot = magnitude_to_db(abs(np.mean(Ptot, axis=2)), parameters['receiver_gain'])

        # output = inverse_RTI(parameters, Pinc, Ptot, inverse_RTI_matrix)
        output = q.get()
        output = output.reshape(parameters['pixel_size'])
        ln.set_data(output)

        now = time.time()
        print(f"{(now - parameters['time']):.4f}s")
        parameters['time'] = now

        return [ln]

    fontdict = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 10,
                }

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.axis('off')

    ln = plt.imshow(np.zeros(parameters['pixel_size']), vmin=0, vmax=1, extent=[0+parameters['grid_resolution']/2, parameters['doi_size'] -
                    parameters['grid_resolution']/2, 0+parameters['grid_resolution']/2, parameters['doi_size']-parameters['grid_resolution']/2], cmap='jet')

    for i in range(parameters['num_devices']):
        plt.scatter(parameters['device_coordinates'][0][i], parameters['device_coordinates'][1][i], c='tan', s=200)
        plt.text(parameters['device_coordinates'][0][i], parameters['device_coordinates'][1][i], s=i+1, fontdict=fontdict, va='center', ha='center')

    anim = animation.FuncAnimation(fig, update, fargs=(parameters, signal, devices, Pinc, inverse_RTI_matrix,), interval=100)
    plt.show()


def data_processing(q, parameters, signal, devices, Pinc, inverse_RTI_matrix):
    while True:
        Ptot = data_collection_once(parameters, signal, devices)
        Ptot = magnitude_to_db(abs(np.mean(Ptot, axis=2)), parameters['receiver_gain'])

        output = inverse_RTI(parameters, Pinc, Ptot, inverse_RTI_matrix)

        q.put(output)


def output_visualization(parameters, signal, devices, Pinc, inverse_RTI_matrix):
    q = Queue()

    p1 = Process(target=data_processing, args=(q, parameters, signal, devices, Pinc, inverse_RTI_matrix,))
    p2 = Process(target=image_display, args=(q, parameters, signal, devices, Pinc, inverse_RTI_matrix,))
    p1.start()
    p2.start()

    p1.join()
    p2.join()


def inverse_RTI_algo(parameters, signal, devices):
    Pinc = np.mean([data_collection_once(parameters, signal, devices) for _ in range(5)], axis=0)
    Pinc = magnitude_to_db(abs(np.mean(Pinc, axis=2)), parameters['receiver_gain'])
    Pinc = Pinc[~np.eye(Pinc.shape[0], dtype=bool)].reshape(-1, 1)

    inverse_RTI_matrix = inverse_RTI_preparation(parameters)
    output_visualization(parameters, signal, devices, Pinc, inverse_RTI_matrix)
