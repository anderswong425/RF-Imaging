from Pluto import Pluto

import curses
import numpy as np
import os
import time
import threading
from multiprocessing import Process, Queue
from scipy.io import savemat


from skimage.restoration import denoise_tv_chambolle
from shapely.geometry import LineString, Point

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def timing_decorator(func):
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}() took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def export_to_mat(arr, name='test'):
    savemat(name + '.mat', {name: arr})


# data collection functions
def generate_signal():
    x_int = np.zeros(1000)
    x_degrees = x_int*360/4.0 + 45  # 45, 135, 225, 315 degrees
    x_radians = x_degrees*np.pi/180.0  # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians)  # this produces our QPSK complex symbols
    signal = x_symbols

    return signal


def init_devices(parameters):
    devices = []
    flag = False
    for i in parameters['device_indices']:
        try:
            devices.append(Pluto(i, parameters))
        except:
            print(f'Pluto{i} not found!')
            flag = True

    if flag:
        quit()
    else:
        return devices


def magnitude_to_dB(MRx, gain):
    PRx = 7 + 20 * np.log10(MRx/1000) - gain
    return PRx


def receive_concurrently(devices):
    '''
    To eliminate the data collection time, all devices will receive data at the same time with multithreading.
    '''
    threads = []
    for device in devices:
        threads.append(threading.Thread(target=device.receive))

    for thread in threads:  # Start all threads
        thread.start()

    for thread in threads:  # Wait for all of them to finish
        thread.join()


def data_collection_once(parameters, signal, devices):
    '''
    As we only care the RSSI, we take mean across the samples of each link, and 
    then use the magnitude to calculate the RSSI.

    The received samples from the same transceiver is dropped.

    Returns a numpy array with shape ((num_devices*(num_devices-1)), 1).

    '''

    dataset = np.zeros((parameters['num_devices'], parameters['num_devices']), dtype=float)

    for tx in range(parameters['num_devices']):
        devices[tx].transmit(signal)
        receive_concurrently(devices)
        devices[tx].stop_transmit()

        for rx in range(parameters['num_devices']):
            dataset[tx][rx] = abs(np.mean(devices[rx].data))

    dataset = magnitude_to_dB(dataset, parameters['receiver_gain'])
    dataset = dataset[~np.eye(dataset.shape[0], dtype=bool)].reshape(-1, 1)

    return dataset


def get_device_coordinates(parameters):
    doi_size = parameters['doi_size']
    num_deivces = len(parameters['device_indices'])

    line = LineString(((-doi_size/2, -doi_size/2), (doi_size/2, -doi_size/2), (doi_size/2, doi_size/2), (-doi_size/2, doi_size/2), (-doi_size/2, -doi_size/2)))

    distances = np.linspace(0, line.length, num_deivces+1)

    points = [line.interpolate(distance) for distance in distances[:-1]]

    coordinates = [[round(point.x, 2), round(point.y, 2)] for point in points]

    return np.array(coordinates)


def get_grid_coordinates(parameters):
    start = -parameters['doi_size']/2 + parameters['doi_size']/(2*parameters['resolution'][0])
    end = abs(start)

    x = np.linspace(start, end, parameters['resolution'][0])

    y = np.linspace(start, end, parameters['resolution'][1])

    grid_coordinates_x, grid_coordinates_y = np.meshgrid(x, y)

    return grid_coordinates_x, grid_coordinates_y


def result_visualization(parameters, image=None, title=None, show_coordinate=False, show_device=True):
    '''
    This function initialize the display for the real time visualization.
    It can also be used to visualize a single image reconstruction result. 

    '''
    if parameters['normailze'] == True:
        if image is not None:
            image = image/image.max()
            image[image < parameters['threshold']] = 0

    fig, ax = plt.subplots(figsize=(11, 8))

    if show_coordinate == False:
        plt.axis('off')
    else:
        plt.grid(False)

    im = plt.imshow(np.zeros(parameters['resolution']), vmin=0, vmax=1, cmap='jet',
                    extent=[-parameters['doi_size']/2, parameters['doi_size']/2, -parameters['doi_size']/2, parameters['doi_size']/2])

    fig.colorbar(im, fraction=0.1, pad=0.1)
    plt.tight_layout(pad=5)

    # add devices display on the plot
    if show_device == True:
        for i in range(parameters['num_devices']):
            plt.text(*parameters['device_coordinates'][i], s=f'{i+1:02d}', va='baseline', ha='center',
                     fontdict={'family': 'serif', 'color':  'black', 'weight': 'bold', 'size': 11},
                     bbox=dict(facecolor='orange', edgecolor='black', boxstyle="circle,pad=0.5"))

    if image is not None:
        im.set_data(image)
        if parameters['normailze'] == True:
            im.set_clim(vmin=0, vmax=1)
        else:
            im.set_clim(vmin=0, vmax=image.max())

        if title is not None:

            fig.suptitle(title, fontsize=16)

    return fig, im


def real_time_visualization(parameters, signal, devices, processing_func, denoising_func=None):
    '''
    This function is for real-time display using matplotlib.animation.
    The data_processing func will collect data repeatedly and handle it with the processing_func,
    while image_display handles how the window looks like.

    To improve the execution speed, these two functions are seperated to different processes.
    '''
    def data_processing(q, parameters, signal, devices, Pinc):
        i = 0
        screen = curses.initscr()
        curses.curs_set(0)

        while True:

            start = time.monotonic()
            Ptot = data_collection_once(parameters, signal, devices)
            screen.addstr(0, 0, f'Data collection: {(time.monotonic()-start)*1000:.0f}ms\n')

            # np.save(f'test/Ptot{i}.npy', Ptot)

            start = time.monotonic()
            output = processing_func(parameters, Pinc, Ptot)
            screen.addstr(1, 0, f'Imaging reconstruction: {(time.monotonic()-start)*1000:.2f}ms\n')

            if denoising_func is not None:
                start = time.monotonic()
                output = denoising_func(output, weight=parameters['denoising_weight'])
                screen.addstr(2, 0, f'Denoising: {(time.monotonic()-start)*1000:.2f}ms\n')

            if parameters['normailze'] == True:
                # output = output/output.max()
                output[output < parameters['threshold']] = 0

            if i == 0:
                process_start = time.monotonic()
            else:
                screen.addstr(4, 0, f'Average acquisition time of {i} frames: {(time.monotonic()-process_start)*1000/(i):.0f}ms\n\n')

            screen.refresh()
            i = i+1
            q.put(output)

    def image_display(q, parameters, signal, devices, Pinc):

        def update(frame, *fargs):
            parameters, signal, devices, Pinc = fargs
            output = q.get()
            im.set_data(output)
            im.set_clim(vmin=0, vmax=np.max(im.get_array()))
            im.set_clim(vmax=0.08)

            return [im]

        fig, im = result_visualization(parameters)
        anim = animation.FuncAnimation(fig, update, fargs=(parameters, signal, devices, Pinc,), interval=10, cache_frame_data=False)

        fig.canvas.mpl_connect('close_event', close_event)
        plt.show()

    def close_event(event):
        p1.terminate()

    Pinc = np.mean([data_collection_once(parameters, signal, devices) for _ in range(5)], axis=0)

    # for testing using saved Pinc
    if parameters['saved_Pinc'] == True:
        Pinc = np.load('Pinc.npy')
    else:
        np.save('Pinc.npy', Pinc)

    parameters['flag'] = 0

    q = Queue()
    p1 = Process(target=data_processing, args=(q, parameters, signal, devices, Pinc,))
    p2 = Process(target=image_display, args=(q, parameters, signal, devices, Pinc,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
