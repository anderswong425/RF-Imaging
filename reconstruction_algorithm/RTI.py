import numpy as np


def RTI(parameters, Pinc, Ptot):
    def calculate_distance(point1, point2):
        return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

    def RTI_preparation(parameters):
        device_xx, device_yy = parameters['device_coordinates'].T

        grid_xx, grid_yy = parameters['grid_coordinates']

        dist_txrx = np.zeros((parameters['num_devices'], parameters['num_devices']))
        for tx in range(parameters['num_devices']):
            for rx in range(parameters['num_devices']):
                dist_txrx[tx][rx] = calculate_distance((device_xx[tx], device_yy[tx]), (device_xx[rx], device_yy[rx]))

        dist_grid2device = np.zeros((*parameters['resolution'], parameters['num_devices']))
        for y in range(parameters['resolution'][1]):
            for x in range(parameters['resolution'][0]):
                for device in range(parameters['num_devices']):
                    dist_grid2device[x][y][device] = (calculate_distance((grid_xx[x][y], grid_yy[x][y]), (device_xx[device], device_yy[device])))

        F_RTI = np.zeros(((parameters['num_devices'])*(parameters['num_devices']-1), *parameters['resolution']))

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
        RTI_matrix = np.linalg.solve((np.matmul(F_RTI.T, F_RTI) + parameters['alpha'] * np.identity((parameters['resolution'][0]**2))),  F_RTI.T)

        return RTI_matrix

    Pryt = Pinc - Ptot
    if not parameters['flag']:

        parameters['G'] = RTI_preparation(parameters)

        parameters['flag'] = True

    output = np.matmul(parameters['G'], Pryt)

    output /= output.max()

    output[output < 0] = 0

    return np.rot90(output.reshape(parameters['resolution'], order='F'))
