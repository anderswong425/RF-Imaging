from scipy.sparse.linalg import lsqr
import numpy as np
import time
from tqdm import tqdm
from scipy.linalg import solve
import pathlib
from functions import *
from xPRA import get_device_coordinates, get_grid_coordinates
import scipy
from shapely.geometry import LineString, Point
from scipy.special import hankel1, jv


def timing_decorator(num_runs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            total_time = 0
            for i in tqdm(range(num_runs)):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                total_time += end - start
            avg_time = total_time / num_runs
            print(f"Avg execution time over {num_runs} runs: {avg_time:.3f} seconds")
            return result
        return wrapper
    return decorator


num_runs = 100


def xPRA_preparation(parameters):
    device_coordinates = get_device_coordinates(parameters)

    grid_coordinates_x, grid_coordinates_y = get_grid_coordinates(parameters)

    grid_coordinates_x, grid_coordinates_y = get_grid_coordinates(parameters)
    xr, xpr = np.meshgrid(device_coordinates[:, 0], grid_coordinates_x.T.reshape(-1))
    yr, ypr = np.meshgrid(device_coordinates[:, 1], grid_coordinates_y.T.reshape(-1)[::-1])

    distRxRn = np.sqrt((xr-xpr)**2+(yr-ypr)**2).T

    Zryt = ((1j*np.pi*parameters['cellrad'] / (2*parameters['k0'])) *
            jv(1, parameters['k0']*parameters['cellrad']) *
            hankel1(0, parameters['k0']*distRxRn.T)).T

    dist_txrx = np.zeros((parameters['num_devices'], parameters['num_devices']))
    for tx in range(parameters['num_devices']):
        for rx in range(parameters['num_devices']):
            dist_txrx[tx][rx] = calculate_distance(device_coordinates[tx], device_coordinates[rx])

    E_d = (1j/4)*hankel1(0, parameters['k0']*dist_txrx)
    E_inc = (1j/4)*hankel1(0, parameters['k0']*distRxRn)
    Fryt = np.zeros((parameters['num_devices']*(parameters['num_devices']-1), parameters['pixel_size'][0]**2), dtype=complex)

    idx = 0
    for tx in range(parameters['num_devices']):
        for rx in range(parameters['num_devices']):
            if tx != rx:
                Fryt[idx] = ((parameters['k0']**2)*((Zryt[rx, :]*(E_inc[tx, :]))/(E_d[rx][tx])))

                idx += 1

    FrytB = np.concatenate((Fryt.real, -Fryt.imag), axis=1)

    FrytBat = FrytB.T @ FrytB

    return FrytB, FrytBat


@timing_decorator(num_runs)
def xPRA(parameters, FrytB, Pinc, Ptot):

    Pryt = (Ptot-Pinc)/(20*np.log10(np.exp(1)))

    lambda_max = np.linalg.norm((FrytB.T @ Pryt), ord=2)

    Oimag = (np.linalg.solve(FrytB.T @ FrytB + lambda_max * parameters['alpha'] * np.identity(FrytB.shape[1]), FrytB.T) @ Pryt)[parameters['pixel_size'][0]**2:]

    epr = 4*np.pi*(Oimag*0.5)/parameters['wavelength']

    epr[epr < 0] = 0

    epr = epr.reshape(parameters['pixel_size'], order='F')

    return epr


def xPRA_preparation_test(parameters):
    device_coordinates = get_device_coordinates(parameters)

    grid_coordinates_x, grid_coordinates_y = get_grid_coordinates(parameters)

    grid_coordinates_x, grid_coordinates_y = get_grid_coordinates(parameters)
    xr, xpr = np.meshgrid(device_coordinates[:, 0], grid_coordinates_x.T.reshape(-1))
    yr, ypr = np.meshgrid(device_coordinates[:, 1], grid_coordinates_y.T.reshape(-1)[::-1])

    distRxRn = np.sqrt((xr-xpr)**2+(yr-ypr)**2).T

    Zryt = ((1j*np.pi*parameters['cellrad'] / (2*parameters['k0'])) *
            jv(1, parameters['k0']*parameters['cellrad']) *
            hankel1(0, parameters['k0']*distRxRn.T)).T

    dist_txrx = np.zeros((parameters['num_devices'], parameters['num_devices']))
    for tx in range(parameters['num_devices']):
        for rx in range(parameters['num_devices']):
            dist_txrx[tx][rx] = calculate_distance(device_coordinates[tx], device_coordinates[rx])

    E_d = (1j/4)*hankel1(0, parameters['k0']*dist_txrx)
    E_inc = (1j/4)*hankel1(0, parameters['k0']*distRxRn)
    Fryt = np.zeros((parameters['num_devices']*(parameters['num_devices']-1), parameters['pixel_size'][0]**2), dtype=complex)

    idx = 0
    for tx in range(parameters['num_devices']):
        for rx in range(parameters['num_devices']):
            if tx != rx:
                Fryt[idx] = ((parameters['k0']**2)*((Zryt[rx, :]*(E_inc[tx, :]))/(E_d[rx][tx])))

                idx += 1

    # FrytB = np.concatenate((Fryt.real, -Fryt.imag), axis=1)
    FrytB = -Fryt.imag

    FrytBat = FrytB.T @ FrytB

    return FrytB, FrytBat


@timing_decorator(num_runs)
def xPRA_test(parameters, FrytB, FrytBat, Pinc, Ptot):
    Pryt = (Ptot-Pinc)/(20*np.log10(np.exp(1)))

    lambda_max = np.linalg.norm((FrytB.T @ Pryt), ord=2)

    Oimag = (np.linalg.solve(FrytBat + lambda_max * parameters['alpha'] * np.identity(FrytB.shape[1]), FrytB.T) @ Pryt)

    epr = 4*np.pi*(Oimag*0.5)/parameters['wavelength']

    epr[epr < 0] = 0

    epr = epr.reshape(parameters['pixel_size'], order='F')

    return epr


parameters = {}
parameters['time'] = time.monotonic()

parameters['num_devices'] = 20
parameters['device_indices'] = [x+1 for x in range(parameters['num_devices'])]

# device parameters
parameters['sample_rate'] = 1e6  # Hz
parameters['num_samples'] = 100  # number of samples per call to rx()
parameters['center_freq'] = 2.4e9  # Hz
parameters['bandwidth'] = 100  # Hz
parameters['transmitter_attenuation'] = 0  # dB
parameters['receiver_gain'] = 30  # dB
parameters['wavelength'] = 3e8/parameters['center_freq']

# imaging parameters
parameters['doi_size'] = 3
parameters['detection_size'] = 0.1

parameters['alpha'] = 3  # 1e2
parameters['denoising_weight'] = 0.2
parameters['pixel_size'] = (60, 60)

parameters['k0'] = 2*np.pi/parameters['wavelength']
parameters['cellrad'] = parameters['doi_size']/(parameters['pixel_size'][0]*np.sqrt(np.pi))


FrytB, FrytBat = xPRA_preparation(parameters)
path = 'result'
test_files = sorted(list(pathlib.Path(path).glob('*.npy')))

Pinc = np.load(test_files.pop(0))
Pinc = magnitude_to_db(abs(np.mean(np.squeeze(Pinc), axis=2)), parameters['receiver_gain'])
Pinc = Pinc[~np.eye(Pinc.shape[0], dtype=bool)].reshape(-1, 1)


Ptot = np.load(test_files.pop(0))
Ptot = magnitude_to_db(abs(np.mean(np.squeeze(Ptot), axis=2)), parameters['receiver_gain'])
Ptot = Ptot[~np.eye(Ptot.shape[0], dtype=bool)].reshape(-1, 1)


A = xPRA(parameters, FrytB, Pinc, Ptot)

FrytB, FrytBat = xPRA_preparation_test(parameters)

B = xPRA_test(parameters, FrytB, FrytBat, Pinc, Ptot)

MSE = np.mean((A - B)**2)
print(f'{MSE=:.4f}')
plt.subplot(1, 2, 1)
plt.imshow(A, cmap='jet')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(B, cmap='jet')
plt.title('Impoved')
plt.show()
