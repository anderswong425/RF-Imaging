# from bokeh.io import output_file, show
# from bokeh.plotting import figure
# from bokeh.layouts import gridplot, row, column
# from bokeh.models.widgets import Tabs, Panel
# from bokeh.models import Div

# def plot_dataset(parameters, dataset, same_scale=False):
#     print('Plotting the constellation diagram of collected data...')

#     if same_scale:
#         lim = 0
#         for i in range(parameters['num_devices']):
#             for j in range(parameters['num_devices']):
#                 max = np.max(np.abs(dataset[i][j]))
#                 if max > lim:
#                     lim = max
#         lim *= 1.1

#     fig, axs = plt.subplots(parameters['num_devices'], parameters['num_devices'], figsize=(10, 8))
#     fig.text(0.5, 0.04, 'In-phase', ha='center', fontsize=12)
#     fig.text(0.04, 0.5, 'Quadrature', va='center', rotation='vertical', fontsize=12)
#     fig.text(0.07, 0.9, 'Tx\Rx', fontsize=14)

#     fig.suptitle(f"Constellation of Received Samples", fontweight="bold")
#     for i in range(parameters['num_devices']):
#         for j in range(parameters['num_devices']):
#             axs[i, j].clear()
#             axs[i, 0].set_ylabel(f'Pluto{i+1}', fontsize=12)
#             axs[0, j].set_title(f'Pluto{j+1}', fontsize=12)
#             axs[i, j].grid(True)
#             axs[i, j].plot(dataset[i][j].real, dataset[i][j].imag, 'bo')

#             if not same_scale:
#                 lim = np.max(np.abs(dataset[i][j])) * 1.1

#             axs[i, j].set_xlim([-lim, lim])
#             axs[i, j].set_ylim([-lim, lim])
#             axs[i, j].set_box_aspect(1)

#         # plt.savefig(dir + f"\constellation_plot{iter+1}.png")

#         # if iter+1 != num_iter:
#         #     plt.show(block=False)
#         #     fig.canvas.draw()
#         #     fig.canvas.flush_events()
#         # else:
#     plt.show()

# def channel_estimation():
#     magnitude_estimations = []
#     phase_estimations = []
#     for iter in range(num_iter):
#         for i in range(num_devices):
#             for j in range(num_devices):

#                 magnitude_estimation = np.divide(np.abs(averages[iter][i][j]), np.abs(signal[:4]))
#                 magnitude_estimation = np.average(magnitude_estimation)
#                 magnitude_estimations.append(magnitude_estimation)

#                 phase_estimation = np.arctan((averages[iter][i][j].imag-signal[:4].imag)/(averages[iter][i][j].real-signal[:4].real))
#                 phase_estimation = np.degrees(phase_estimation)
#                 phase_estimation = np.average(phase_estimation)
#                 phase_estimations.append(phase_estimation)

#     magnitude_estimations = np.reshape(magnitude_estimations, (-1, num_devices, num_devices)).transpose()
#     magnitude_estimations = np.swapaxes(magnitude_estimations, 0, 1)

#     phase_estimations = np.reshape(phase_estimations, (-1, num_devices, num_devices)).transpose()
#     phase_estimations = np.swapaxes(phase_estimations, 0, 1)

#     return magnitude_estimations, phase_estimations

# def plot_channel_estimation(estimation, title):
#     print('\nPlotting the channel estimation of collected data...')
#     fig, axs = plt.subplots(num_devices, num_devices, figsize=(16, 8))
#     fig.suptitle(f"{title}", fontweight="bold")
#     fig.text(0.5, 0.0, 'Interation', ha='center', fontsize=12)
#     for i in range(num_devices):
#         for j in range(num_devices):
#             axs[i, j].set_title(f"Tx: Pluto{i+1}, Rx: Pluto{j+1}", fontsize=12)
#             axs[i, j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
#             axs[i, j].set_box_aspect(1/3)
#             axs[i, j].grid(True)
#             axs[i, j].margins(x=0)
#             axs[i, j].plot(range(1, num_iter+1), estimation[i][j], 'b-')

#     plt.tight_layout()
#     plt.savefig(dir + f"\channel_estimation_{title}.png")


# def data_rolling():
#     averages = []
#     for iter in range(num_iter):
#         for i in range(num_devices):
#             for j in range(num_devices):
#                 four_groups = np.split(dataset[iter][i][j], num_samps//4)
#                 four_groups = np.transpose(four_groups)
#                 average = np.average(four_groups, axis=1)

#                 average_angle = np.angle(average, deg=True)
#                 roll_num = 4 - (np.abs(average_angle - 45)).argmin()
#                 dataset[iter][i][j] = np.roll(dataset[iter][i][j], roll_num)

#                 four_groups = np.split(dataset[iter][i][j], num_samps//4)
#                 four_groups = np.transpose(four_groups)
#                 average = np.average(four_groups, axis=1)
#                 averages.append(average)

#     averages = np.reshape(averages, (num_iter, num_devices, num_devices, 4))
#     return dataset, averages
