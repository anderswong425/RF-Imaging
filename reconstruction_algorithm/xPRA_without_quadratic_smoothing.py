import numpy as np
from scipy.special import hankel1, jv


'''
The original implementation of the xPRA algorithm.
The result is not as good as the one with quadratic smoothing. 
The calculation is more complicated which will take longer time but it is a simplified version.
'''


def xPRA(parameters, Pinc, Ptot):
    def xPRA_preparation(parameters):
        def calculate_distance(point1, point2):
            return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

        device_coordinates = parameters['device_coordinates']
        grid_coordinates_x, grid_coordinates_y = parameters['grid_coordinates']

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
        Fryt = np.zeros((parameters['num_devices']*(parameters['num_devices']-1), parameters['resolution'][0]**2), dtype=complex)

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

    Pryt = (Ptot-Pinc)/(20*np.log10(np.exp(1)))

    if not parameters['flag']:
        FrytB, FrytBat = xPRA_preparation(parameters)

        lambda_max = np.linalg.norm((FrytB.T @ Pryt), ord=2)
        parameters['G'] = np.linalg.solve(FrytBat + lambda_max * parameters['alpha'] * np.identity(FrytB.shape[1]), FrytB.T)

        parameters['flag'] = True

    Oimag = parameters['G'] @ Pryt
    epr = 4*np.pi*(Oimag*0.5)/parameters['wavelength']

    epr[epr < 0] = 0

    return epr.reshape(parameters['resolution'], order='F')


