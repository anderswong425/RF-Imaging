from functions import *
from inverse_RTI import inverse_RTI_preparation, output_visualization


def main():
    parameters = {}

    parameters['time'] = time.time()
    parameters['num_devices'] = 20
    parameters['device_indices'] = [x+1 for x in range(parameters['num_devices'])]

    # device parameters
    parameters['sample_rate'] = 1e6  # Hz
    parameters['num_samples'] = 100  # number of samples per call to rx()
    parameters['center_freq'] = 2.35e9  # Hz
    parameters['bandwidth'] = 10  # Hz
    parameters['transmitter_attenuation'] = 0  # dB
    parameters['receiver_gain'] = 30  # dB

    # imaging parameters
    parameters['doi_size'] = 1
    parameters['alpha'] = 1e2
    parameters['grid_resolution'] = 0.05
    parameters['detection_size'] = 0.1
    parameters['pixel_size'] = (int(parameters['doi_size']/parameters['grid_resolution']), int(parameters['doi_size']/parameters['grid_resolution'])) 

    signal = generate_signal()

    devices = init_devices(parameters)

    Pinc = np.mean([data_collection_once(parameters, signal, devices) for _ in range(5)], axis=0)

    Pinc = magnitude_to_db(abs(np.mean(Pinc, axis=2)), parameters['receiver_gain'])
    Pinc = Pinc[~np.eye(Pinc.shape[0], dtype=bool)].reshape(-1, 1)

    inverse_RTI_matrix = inverse_RTI_preparation(parameters)

    output_visualization(parameters, signal, devices, Pinc, inverse_RTI_matrix)


if __name__ == '__main__':
    main()
