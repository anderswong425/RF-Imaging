from functions import *
from shapely.geometry import LineString, Point
from scipy.special import hankel1, jv


def get_device_coordinates(parameters):
    doi_size = parameters['doi_size']
    num_deivces = len(parameters['device_indices'])

    # line = LineString(((0.0, 0.0), (0.0, doi_size), (doi_size, doi_size), (doi_size, 0.0), (0.0, 0.0)))
    line = LineString(((-doi_size/2, -doi_size/2), (doi_size/2, -doi_size/2), (doi_size/2, doi_size/2), (-doi_size/2, doi_size/2), (-doi_size/2, -doi_size/2)))

    distances = np.linspace(0, line.length, num_deivces+1)

    points = [line.interpolate(distance) for distance in distances[:-1]]

    coordinates = [[round(point.x, 2), round(point.y, 2)] for point in points]

    return np.array(coordinates)


def get_grid_coordinates(parameters):
    start = -parameters['doi_size']/2 + parameters['grid_resolution']/2
    end = abs(start)

    x = np.linspace(start, end, parameters['pixel_size'][0])

    y = np.linspace(start, end, parameters['pixel_size'][1])

    grid_coordinates_x, grid_coordinates_y = np.meshgrid(x, y)

    return grid_coordinates_x, grid_coordinates_y


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

    return FrytB


def xPRA(parameters, Pinc, Ptot):
    FrytB = xPRA_preparation(parameters)
    Pryt = (Ptot-Pinc)/(20*np.log10(np.exp(1)))

    lambda_max = np.linalg.norm(np.matmul(FrytB.T, Pryt), ord=2)
    Oimag = np.matmul(np.linalg.solve((np.matmul(FrytB.T, FrytB) + lambda_max * parameters['alpha'] * np.identity(FrytB.shape[1])),  FrytB.T), Pryt)[parameters['pixel_size'][0]**2:]
    epr = 4*np.pi*(Oimag*0.5)/parameters['wavelength']

    epr[epr < 0] = 0

    epr = epr.reshape(parameters['pixel_size'], order='F')

    return epr
