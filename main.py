from functions import *
from inverse_RTI import inverse_RTI_algo
from xPRA import xPRA_preparation, xPRA


def main():
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
    parameters['denoising_weight'] = 0.1
    parameters['pixel_size'] = (60, ) * 2  # NxN square matrix

    parameters['k0'] = 2*np.pi/parameters['wavelength']
    parameters['cellrad'] = parameters['doi_size']/(parameters['pixel_size'][0]*np.sqrt(np.pi))

    signal = generate_signal()

    devices = init_devices(parameters)

    # inverse_RTI_algo(parameters, signal, devices)
    real_time_visualization(parameters, signal, devices, xPRA_preparation, xPRA)


if __name__ == '__main__':
    main()
