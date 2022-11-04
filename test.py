from functions import *
from inverse_RTI import inverse_RTI_preparation, output_visualization


def main():
    parameters = {}

    parameters['time'] = time.strftime('%d%b%H%M', time.localtime())
    parameters['doi_size'] = 1.5
    parameters['alpha'] = 1e2

    parameters['num_devices'] = 20
    parameters['device_indices'] = [x+1 for x in range(parameters['num_devices'])]

    parameters['sample_rate'] = 1e6  # Hz
    parameters['num_samples'] = 1000  # number of samples per call to rx()
    parameters['center_freq'] = 2.35e9  # Hz 2.4e9
    parameters['bandwidth'] = 10  # Hz
    parameters['transmitter_attenuation'] = 0  # dB
    parameters['receiver_gain'] = 40  # dB
    parameters['grid_resolution'] = 0.05
    parameters['detection_size'] = 0.1

    signal = generate_signal()

    devices = init_devices(parameters)

    Pinc = data_collection_once(parameters, signal, devices)
    Pinc = magnitude_to_db(abs(np.mean(Pinc, axis=2)), parameters['receiver_gain'])
    Pinc = Pinc[~np.eye(Pinc.shape[0], dtype=bool)].reshape(-1, 1)

    RTI_matrix = inverse_RTI_preparation(parameters)

    output_visualization(parameters, signal, devices, Pinc, RTI_matrix)


if __name__ == '__main__':
    main()
