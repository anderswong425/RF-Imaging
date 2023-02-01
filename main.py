from functions import *
from inverse_RTI import inverse_RTI_algo


def main():
    parameters = {}

    parameters['time'] = time.time()
    parameters['num_devices'] = 20
    parameters['device_indices'] = [x+1 for x in range(parameters['num_devices'])]

    # device parameters
    parameters['sample_rate'] = 1e6  # Hz
    parameters['num_samples'] = 100  # number of samples per call to rx()
    parameters['center_freq'] = 2.4e9  # Hz
    parameters['bandwidth'] = 10  # Hz
    parameters['transmitter_attenuation'] = 0  # dB
    parameters['receiver_gain'] = 40  # dB

    # imaging parameters
    parameters['doi_size'] = 3
    parameters['alpha'] = 1e2  # 1e2
    parameters['grid_resolution'] = 0.1
    parameters['detection_size'] = 0.1
    parameters['pixel_size'] = (int(parameters['doi_size']/parameters['grid_resolution']), int(parameters['doi_size']/parameters['grid_resolution']))

    parameters['eterm'] = 1

    signal = generate_signal()

    devices = init_devices(parameters)

    inverse_RTI_algo(parameters, signal, devices)


if __name__ == '__main__':
    main()
