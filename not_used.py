from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot, row, column
from bokeh.models.widgets import Tabs, Panel
from bokeh.models import Div


# data processing functions
def create_result_folder(dir):
    if os.path.exists(dir):
        #     shutil.rmtree(dir)
        pass
    else:
        os.makedirs(dir)


def data_collection(parameters, signal, devices):
    print('\nCollecting data from the devices...')
    dataset = []
    time_sequence = [time.time()]

    for i in tqdm(range(parameters['num_iter'])):
        dataset.append(data_collection_once(parameters, signal, devices))
        time_sequence.append(time.time())

    for i in range(1, len(time_sequence)):
        time_sequence[i] = round((time_sequence[i] - time_sequence[0]), 3)
    time_sequence.pop(0)

    print(f"Data collection of {parameters['num_iter']} sets of data with {len(parameters['device_indices'])} devices is finished in {time_sequence[-1]:.3f}s")

    dir = f"result\\{parameters['time']}"
    create_result_folder(dir)

    np.save(f"{dir}\\dataset", dataset)
    np.save(f"{dir}\\time_sequence", time_sequence)

    return dataset, time_sequence


def friis_transmission_equation(Pt, Gt, Gr, d, freq):
    lamda = 3e8/freq
    Pr = Pt + Gt + Gr + 20*np.log10((lamda/(4*np.pi*d)))

    return Pr


def constellation_plot(parameters, dataset, same_scale=False):
    if same_scale == True:
        lim = 0
        for i in range(parameters['num_iter']):
            for tx in range(len(parameters['device_indices'])):
                for rx in range(len(parameters['device_indices'])):
                    max = np.max(np.abs(dataset[i][tx][rx]))
                    if max > lim:
                        lim = max
        lim *= 1.1

    tabs = []
    for i in tqdm(range(parameters['num_iter'])):
        figures = []
        for tx in range(len(parameters['device_indices'])):
            for rx in range(len(parameters['device_indices'])):
                if same_scale == False:
                    lim = np.max(np.abs(dataset[i][tx][rx])) * 1.3

                fig = figure(background_fill_color="#fafafa", x_range=(-lim, lim), y_range=(-lim, lim), title=f"Tx{parameters['device_indices'][tx]} Rx{parameters['device_indices'][rx]}")
                fig.scatter(dataset[i][tx][rx].real, dataset[i][tx][rx].imag, size=5, color="#53777a")

                figures.append(fig)

        grid = gridplot(figures, ncols=len(parameters['device_indices']), width=210, height=210)

        tabs.append(Panel(child=grid, title=str(i+1)))

    print('\nGenerating the dashboard...', end=' ')
    plot = Tabs(tabs=tabs)

    return plot


def magnitude_plot(parameters, dataset, time_sequence):
    figures = []
    x = time_sequence

    for tx in tqdm(range(len(parameters['device_indices'])), disable=True):
        for rx in range(len(parameters['device_indices'])):
            y = []
            for i in range(parameters['num_iter']):
                y.append(np.average(np.abs(dataset[i][tx][rx])))

            fig = figure(background_fill_color="#fafafa", y_range=(0, max(y)*1.4), title=f"Tx{parameters['device_indices'][tx]} Rx{parameters['device_indices'][rx]}")
            fig.line(x, y, color="firebrick")
            fig.circle(x, y, color="firebrick")
            figures.append(fig)

            # if tx != rx:
            #     print(f"\nTx: {(tx+1)} Rx: {(rx+1)} {int(np.average(y))}")

    grid = gridplot(figures, ncols=len(parameters['device_indices']), width=210, height=210)

    title = '''<b><u>Magnitude</u></b>'''

    plot = column(Div(text=title), grid)
    return plot


def plot_dashboard(parameters, dataset, time_sequence, same_scale=False):
    output_file("result.html")

    print(f'\nPreparing for data visualization...')

    constellation = constellation_plot(parameters, dataset, same_scale)
    magnitude = magnitude_plot(parameters, dataset, time_sequence)

    text = f'''<p style='margin-top:150px'></p>
               <p style='margin:0px 10px; font-size: 17px'>Number of samples: {si_format(parameters['num_samples'], precision=0)}</p>
               <p style='margin:0px 10px; font-size: 17px'>Sampling rate: {si_format(parameters['sample_rate'], precision=0)}Hz</p>
               <p style='margin:0px 10px; font-size: 17px'>Bandwidth: {si_format(parameters['bandwidth'], precision=0)}Hz</p>
               <p style='margin:0px 10px; font-size: 17px'>Center Frequency: {si_format(parameters['center_freq'], precision=2)}Hz</p>
               <p style='margin:0px 10px; font-size: 17px'>Transmitter attenuation: {parameters['transmitter_attenuation']} dB</p>
               <p style='margin:0px 10px; font-size: 17px'>Receiver gain: {parameters['receiver_gain']} dB</p>
               <p style='margin:50px 0px 0px 10px; font-size: 17px'>Number of devices: {len(parameters['device_indices'])}</p>
               <p style='margin:0px 10px; font-size: 17px'>Number of iterations: {parameters['num_iter']}</p>
               <p style='margin:0px 10px; font-size: 17px'>Time: {time_sequence[-1]:.3f}s</p>
            '''

    show(row(children=[constellation, Div(text=text), magnitude], sizing_mode="stretch_both"), new='window')
    print('\nIt will be shown on the browser when it is ready.')
    print('It may take a while if the collected dataset is big.')


def plot_non_interative(parameters, dataset, time_sequence):
    x = time_sequence

    fig, axs = plt.subplots(len(parameters['device_indices']), len(parameters['device_indices']))

    for tx in tqdm(range(len(parameters['device_indices'])), disable=True):
        for rx in range(len(parameters['device_indices'])):
            y = []
            for i in range(parameters['num_iter']):
                y.append(np.average(np.abs(dataset[i][tx][rx])))

            axs[tx, rx].plot(x, y, 'b-')
            axs[tx, rx].set_ylim(bottom=0, top=max(y)*2)
            axs[tx, rx].grid()
            axs[tx, rx].set_title(f"Tx{parameters['device_indices'][tx]} Rx{parameters['device_indices'][rx]}", fontsize=30)

            if tx == rx:
                for spine in axs[tx, rx].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(5)
            # else:
            #     print(f"\nTx: {parameters['device_indices'][tx]} Rx: {parameters['device_indices'][rx]} {int(np.average(y))}")

    fig.set_size_inches(100, 100, forward=True)
    parameters_str = f'''
        Number of samples: {si_format(parameters['num_samples'], precision=0)}
        Sampling rate: {si_format(parameters['sample_rate'], precision=0)}Hz
        Bandwidth: {si_format(parameters['bandwidth'], precision=0)}Hz
        Center Frequency: {si_format(parameters['center_freq'], precision=2)}Hz
        Transmitter attenuation: {parameters['transmitter_attenuation']} dB
        Receiver gain: {parameters['receiver_gain']} dB

        Number of devices: {len(parameters['device_indices'])}
        Number of iterations: {parameters['num_iter']}

        Time: {time_sequence[-1]:.3f}s
        '''
    plt.text(0.0, 0.5, parameters_str, fontsize=50, transform=plt.gcf().transFigure)

    dir = f"result\\{parameters['time']}"
    create_result_folder(dir)

    plt.savefig(dir + "\\result.png", dpi=150)
    print(f"Result is saved in '{dir}'")
    os.startfile(dir + "\\result.png")


def get_coordinate(idx, parameters):
    num_devices = len(parameters['device_indices'])
    if idx <= num_devices/4+1:
        x = 0
        y = parameters['size'] * 4 / num_devices * (idx-1)
        theta = 0
    elif idx <= num_devices / 2 + 1:
        x = parameters['size'] * 4 / num_devices * (idx-num_devices/4-1)
        y = parameters['size']
        theta = -np.pi/2
    elif idx <= num_devices * 3/4 + 1:
        x = parameters['size']
        y = parameters['size'] * (1-(idx-1-num_devices/2) * 4 / num_devices)
        theta = np.pi
    else:
        x = parameters['size'] * (1-(idx-1-num_devices*3/4) * 4 / num_devices)
        y = 0
        theta = np.pi/2

    if idx == 1:
        theta = np.pi/4
    elif idx == 1+num_devices/4:
        theta = -np.pi/4
    elif idx == 1+num_devices/4*2:
        theta = -3*np.pi/4
    elif idx == 1+num_devices/4*3:
        theta = 3*np.pi/4

    return [x, y, theta]


def plot_dataset(parameters, dataset, same_scale=False):
    print('Plotting the constellation diagram of collected data...')

    if same_scale:
        lim = 0
        for i in range(parameters['num_devices']):
            for j in range(parameters['num_devices']):
                max = np.max(np.abs(dataset[i][j]))
                if max > lim:
                    lim = max
        lim *= 1.1

    fig, axs = plt.subplots(parameters['num_devices'], parameters['num_devices'], figsize=(10, 8))
    fig.text(0.5, 0.04, 'In-phase', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Quadrature', va='center', rotation='vertical', fontsize=12)
    fig.text(0.07, 0.9, 'Tx\Rx', fontsize=14)

    fig.suptitle(f"Constellation of Received Samples", fontweight="bold")
    for i in range(parameters['num_devices']):
        for j in range(parameters['num_devices']):
            axs[i, j].clear()
            axs[i, 0].set_ylabel(f'Pluto{i+1}', fontsize=12)
            axs[0, j].set_title(f'Pluto{j+1}', fontsize=12)
            axs[i, j].grid(True)
            axs[i, j].plot(dataset[i][j].real, dataset[i][j].imag, 'bo')

            if not same_scale:
                lim = np.max(np.abs(dataset[i][j])) * 1.1

            axs[i, j].set_xlim([-lim, lim])
            axs[i, j].set_ylim([-lim, lim])
            axs[i, j].set_box_aspect(1)

        # plt.savefig(dir + f"\constellation_plot{iter+1}.png")

        # if iter+1 != num_iter:
        #     plt.show(block=False)
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()
        # else:
    plt.show()


def channel_estimation():
    magnitude_estimations = []
    phase_estimations = []
    for iter in range(num_iter):
        for i in range(num_devices):
            for j in range(num_devices):

                magnitude_estimation = np.divide(np.abs(averages[iter][i][j]), np.abs(signal[:4]))
                magnitude_estimation = np.average(magnitude_estimation)
                magnitude_estimations.append(magnitude_estimation)

                phase_estimation = np.arctan((averages[iter][i][j].imag-signal[:4].imag)/(averages[iter][i][j].real-signal[:4].real))
                phase_estimation = np.degrees(phase_estimation)
                phase_estimation = np.average(phase_estimation)
                phase_estimations.append(phase_estimation)

    magnitude_estimations = np.reshape(magnitude_estimations, (-1, num_devices, num_devices)).transpose()
    magnitude_estimations = np.swapaxes(magnitude_estimations, 0, 1)

    phase_estimations = np.reshape(phase_estimations, (-1, num_devices, num_devices)).transpose()
    phase_estimations = np.swapaxes(phase_estimations, 0, 1)

    return magnitude_estimations, phase_estimations


def plot_channel_estimation(estimation, title):
    print('\nPlotting the channel estimation of collected data...')
    fig, axs = plt.subplots(num_devices, num_devices, figsize=(16, 8))
    fig.suptitle(f"{title}", fontweight="bold")
    fig.text(0.5, 0.0, 'Interation', ha='center', fontsize=12)
    for i in range(num_devices):
        for j in range(num_devices):
            axs[i, j].set_title(f"Tx: Pluto{i+1}, Rx: Pluto{j+1}", fontsize=12)
            axs[i, j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            axs[i, j].set_box_aspect(1/3)
            axs[i, j].grid(True)
            axs[i, j].margins(x=0)
            axs[i, j].plot(range(1, num_iter+1), estimation[i][j], 'b-')

    plt.tight_layout()
    plt.savefig(dir + f"\channel_estimation_{title}.png")


def data_rolling():
    averages = []
    for iter in range(num_iter):
        for i in range(num_devices):
            for j in range(num_devices):
                four_groups = np.split(dataset[iter][i][j], num_samps//4)
                four_groups = np.transpose(four_groups)
                average = np.average(four_groups, axis=1)

                average_angle = np.angle(average, deg=True)
                roll_num = 4 - (np.abs(average_angle - 45)).argmin()
                dataset[iter][i][j] = np.roll(dataset[iter][i][j], roll_num)

                four_groups = np.split(dataset[iter][i][j], num_samps//4)
                four_groups = np.transpose(four_groups)
                average = np.average(four_groups, axis=1)
                averages.append(average)

    averages = np.reshape(averages, (num_iter, num_devices, num_devices, 4))
    return dataset, averages


def generate_gain_table(parameters, devices, signal):
    if devices[0].sdr.gain_control_mode_chan0 == 'manual':
        for device in devices:
            device.sdr.gain_control_mode_chan0 = 'slow_attack'

    gain_table = np.zeros((len(devices), len(devices)), np.int32)

    print('Finding the receiver gain for each channel:')
    for itx, tx in enumerate(pbar := tqdm(devices)):
        tx.transmit(signal)
        time.sleep(10)
        for irx, rx in enumerate(devices):
            gain = []
            if itx != irx:
                for _ in range(10):
                    gain.append(rx.sdr._get_iio_attr('voltage0', 'hardwaregain', False))
                gain_table[itx][irx] = np.mean(gain)
        pbar.set_description(f"Tx{itx+1}: {gain_table[itx]}")

        tx.stop_transmit()

    return gain_table


def xRPI_preparation(parameters):
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

    FrytB = Fryt

    FrytBat = FrytB.T @ FrytB

    return FrytB, FrytBat


def xRPI(parameters, FrytB, FrytBat, Pinc, Ptot):

    Pryt = (Ptot-Pinc)/(20*np.log10(np.exp(1)))

    if not parameters['flag']:
        lambda_max = np.linalg.norm((FrytB.T @ Pryt), ord=2)
        parameters['G'] = np.linalg.solve(FrytBat + lambda_max * parameters['alpha'] * np.identity(FrytB.shape[1]), FrytB.T)

        parameters['flag'] = True
    Oimag = (parameters['G'] @ Pryt)[parameters['pixel_size'][0]**2:]

    epr = 4*np.pi*(Oimag*0.5)/parameters['wavelength']

    epr[epr < 0] = 0

    epr = epr.reshape(parameters['pixel_size'], order='F')

    return epr
