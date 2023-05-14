import numpy as np
from scipy.special import hankel1, jv

'''
This is the updated version of xRPI implementation.
Unlike the original, non-simplified version, there is a part of calculation can be completely separated.
It should be used rather than the original one as the performance is much better.
'''


def xRPI(parameters, Pinc, Ptot):
    def xRPI_preparation(parameters):
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
        Fryt = np.zeros((parameters['num_devices']*(parameters['num_devices']-1), parameters['pixel_size'][0]**2), dtype=complex)

        idx = 0
        for tx in range(parameters['num_devices']):
            for rx in range(parameters['num_devices']):
                if tx != rx:
                    Fryt[idx] = ((parameters['k0']**2)*((Zryt[rx, :]*(E_inc[tx, :]))/(E_d[rx][tx])))

                    idx += 1

        return Fryt

    def quadratic_smoothing(parameters, Pryt, FrytB):
        '''
        Reference:
        https://github.com/dsamruddhi/Inverse-Scattering-Problem/blob/master/inverse_problem/regularize.py
        '''
        def difference_operator(m, num_grids, direction):
            d_row = np.zeros((1, num_grids))
            d_row[0, 0] = 1

            if direction == "horizontal":
                d_row[0, 1] = -2
                d_row[0, 2] = 1

            elif direction == "vertical":
                d_row[0, m] = -2
                d_row[0, 2 * m] = 1

            else:
                raise ValueError("Invalid direction value for difference operator")

            rows = list()
            rows.append(d_row)
            for i in range(0, num_grids - 1):
                shifted_row = np.roll(d_row, 1)
                shifted_row[0, 0] = 0
                rows.append(shifted_row)
                d_row = shifted_row

            d = np.vstack([row for row in rows])
            return d

        m = parameters['pixel_size'][0]
        dim = m**2

        Dx = difference_operator(m, dim, 'horizontal')
        Dy = difference_operator(m, dim, 'vertical')

        return np.linalg.inv((Fryt.T @ Fryt) + parameters['alpha'] * (Dx.T @ Dx + Dy.T @ Dy)) @ Fryt.T

    Pryt = (Ptot-Pinc)/(20*np.log10(np.exp(1)))

    if not parameters['flag']:
        Fryt = xRPI_preparation(parameters)
        parameters['G'] = quadratic_smoothing(parameters, Pryt, Fryt)
        parameters['flag'] = True

    chi = (parameters['G'] @ Pryt).imag

    chi[chi < 0] = 0

    return chi.reshape(parameters['pixel_size'], order='F')
