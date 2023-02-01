from functions import *
from inverse_RTI import *
from xPRA import *


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
parameters['wavelength'] = 3e8/parameters['center_freq']


# imaging parameters
parameters['doi_size'] = 3
parameters['alpha'] = 1e2  # 1e2
parameters['grid_resolution'] = 0.1
parameters['detection_size'] = 0.1
parameters['pixel_size'] = (int(parameters['doi_size']/parameters['grid_resolution']), int(parameters['doi_size']/parameters['grid_resolution']))

parameters['eterm'] = 1
parameters['k0'] = 2*np.pi/parameters['wavelength']
parameters['cellrad'] = (np.sqrt(parameters['grid_resolution']**2/np.pi)*2)/2


signal = generate_signal()

# devices = init_devices(parameters)


RTI_matrix = inverse_RTI_preparation(parameters)
FrytB = xPRA_preparation(parameters)

for idx in range(44, 48):
    Pinc = np.load('result/2.npy')
    Ptot = np.load(f'result/{idx}.npy')

    Pinc1 = magnitude_to_db(abs(np.mean(np.squeeze(Pinc), axis=2)), parameters['receiver_gain'])
    Pinc1 = Pinc1[~np.eye(Pinc.shape[0], dtype=bool)].reshape(-1, 1)
    Ptot1 = magnitude_to_db(abs(np.mean(np.squeeze(Ptot), axis=2)), parameters['receiver_gain'])

    Pinc2 = magnitude_to_db(abs(np.mean(np.squeeze(Pinc), axis=2)), parameters['receiver_gain'])
    Pinc2 = Pinc2[~np.eye(Pinc.shape[0], dtype=bool)].reshape(-1, 1)
    Ptot2 = magnitude_to_db(abs(np.mean(np.squeeze(Ptot), axis=2)), parameters['receiver_gain'])
    Ptot2 = Ptot2[~np.eye(Ptot.shape[0], dtype=bool)].reshape(-1, 1)


    inverse_RTI_result = inverse_RTI(parameters, Pinc1, Ptot1, RTI_matrix).reshape(int(parameters['doi_size']/parameters['grid_resolution']), int(parameters['doi_size']/parameters['grid_resolution']))
    xPRA_result = xPRA(parameters, Pinc2, Ptot2)


    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(inverse_RTI_result, cmap='jet')
    # ax2.imshow(xPRA_result, cmap='jet')

    # ax1.set_title('RTI')
    # ax2.set_title('XPRA')
    # fig.set_size_inches(12, 6)
    # fig.canvas.manager.set_window_title(f'Ptot: {idx}.npy')
    # plt.show()
