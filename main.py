from functions import *


def main():
    parameters = {}

    parameters['time'] = time.strftime('%d%b%H%M', time.localtime())
    parameters['doi_size'] = 1.5
    parameters['alpha'] = 1e2
    parameters['num_iter'] = 1,
    parameters['num_devices'] = 20
    parameters['device_indices'] = [x+1 for x in range(parameters['num_devices'])]

    parameters['sample_rate'] = 1e6  # Hz
    parameters['num_samples'] = 1000  # number of samples per call to rx()
    parameters['center_freq'] = 2.35e9  # Hz 2.4e9
    parameters['bandwidth'] = 10  # Hz
    parameters['transmitter_attenuation'] = 0  # dB
    parameters['receiver_gain'] = 40,  # dB
    parameters['grid_resolution'] = 0.05
    parameters['detection_size'] = 0.1

    parameters['gain_table'] = np.load('result\\gain_table.npy')

    with open('result\\parameters.txt', 'w') as data:
        for key, value in parameters.items():
            data.write('%s:%s\n' % (key, value))

    signal = generate_signal()

    devices = init_devices(parameters)

    dataset, time_sequence = data_collection(parameters, signal, devices)

    # plot_dashboard(parameters, dataset, time_sequence, True)
    # magnitude_plot(parameters, dataset, time_sequence)

    # plot_non_interative(parameters, dataset, time_sequence)


if __name__ == '__main__':
    main()
