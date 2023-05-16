from functions import *

from reconstruction_algorithm.xRPI import xRPI
from reconstruction_algorithm.RTI import RTI

from skimage.restoration import denoise_tv_chambolle


def main():
    parameters = {}
    parameters['num_devices'] = 20
    parameters['device_indices'] = [x+1 for x in range(parameters['num_devices'])]

    # device parameters
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
    parameters['resolution'] = (60, ) * 2  # pixek count
    parameters['alpha'] = 100  # 1e2
    parameters['denoising_weight'] = 0.05
    parameters['k0'] = 2*np.pi/parameters['wavelength']
    parameters['cellrad'] = parameters['doi_size']/(parameters['resolution'][0]*np.sqrt(np.pi))

    parameters['device_coordinates'] = get_device_coordinates(parameters)
    parameters['grid_coordinates'] = get_grid_coordinates(parameters)
    parameters['flag'] = False

    signal = generate_signal()
    devices = init_devices(parameters)

    real_time_visualization(parameters, signal, devices, processing_func=xRPI, denoising_func=denoise_tv_chambolle)

    # Pinc = np.load('Pinc.npy')
    # Ptot = data_collection_once(parameters, signal, devices)

    # result = xRPI(parameters, Pinc, Ptot)
    # result_visualization(parameters, result)
    # result_visualization(parameters, denoise_tv_chambolle(result), 'Further Denoise with TV Denoising [weight=0.1]')

    # plt.show()


if __name__ == '__main__':
    main()
