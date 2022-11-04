from functions import *


def main():
    parameters = {
        'sample_rate': 1e6,  # Hz
        'num_samples': 10000,  # number of samples per call to rx()
        'center_freq': 2.35e9,  # Hz 2.4e9
        'bandwidth': 10,  # Hz
        'transmitter_attenuation': 0,  # dB
        'receiver_gain': 0,  # dB

        'size': 1.5,
        'num_iter': 10,
        # 'device_indices': [1, 6, 11, 16],
        'device_indices': [x+1 for x in range(20)],

        'time': time.strftime('%d%b%H%M', time.localtime())
    }

    with open('result\\parameters.txt', 'w') as data:
        for key, value in parameters.items():
            data.write('%s:%s\n' % (key, value))

    signal = generate_signal()

    devices = init_devices(parameters)

    parameters['gain_table'] = generate_gain_table(parameters, devices, signal)
    with open('result\\gain_table.txt', 'w') as data:
        for key, value in parameters.items():
            data.write('%s:%s\n' % (key, value))

    for i in range(22):
        parameters['time'] = time.strftime('%d%b%H%M', time.localtime())
        dataset, time_sequence = data_collection(parameters, signal, devices)
        time.sleep(30)

    # plot_dashboard(parameters, dataset, time_sequence, True)
    # magnitude_plot(parameters, dataset, time_sequence)

    # plot_non_interative(parameters, dataset, time_sequence)


if __name__ == '__main__':
    main()
    # for i in range(22):
    #     main()
    #     time.sleep(30)

    # i = 0
    # while i < 13:
    #     if int(time.strftime('%M', time.localtime())) == 0:
    #         main()
    #         i = i + 1
    #     else:
    #         # print(time.strftime('%d%b%H%M', time.localtime()))
    #         time.sleep(10)
