import adi
import time
import numpy as np


class Pluto:
    def __init__(self, idx, parameters):
        self.idx = idx
        self.sample_rate = parameters['sample_rate']  # Hz
        self.num_samples = parameters['num_samples']  # number of samples per call to rx()
        self.center_freq = parameters['center_freq']  # Hz
        self.bandwidth = parameters['bandwidth']  # Hz
        self.transmitter_attenuation = parameters['transmitter_attenuation']  # dB
        self.receiver_gain = parameters['receiver_gain']  # dB
        self.frequency_offset = 0
        self.data = np.zeros(parameters['num_samples'])

        self.sdr = adi.Pluto(f"ip:192.168.{idx+1}.1")
        # Config Tx
        self.sdr.tx_lo = int(self.center_freq+self.frequency_offset)
        self.sdr.tx_rf_bandwidth = int(self.bandwidth)
        self.sdr.tx_hardwaregain_chan0 = self.transmitter_attenuation
        self.sdr.tx_cyclic_buffer = True  # Enable cyclic buffers
        # Config Rx
        self.sdr.rx_lo = int(self.center_freq+self.frequency_offset)
        self.sdr.rx_rf_bandwidth = int(self.bandwidth)
        self.sdr.rx_buffer_size = int(self.num_samples)
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = self.receiver_gain
        self.sdr.sample_rate = int(self.sample_rate)

        self.sdr.tx_destroy_buffer()
        self.sdr.rx_destroy_buffer()

    def transmit(self, signal):
        tx_data = signal * 2**14  # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1
        self.sdr.tx(tx_data)

    def stop_transmit(self):
        self.sdr.tx_destroy_buffer()

    def receive(self):
        self.sdr.rx_destroy_buffer()
        self.data = self.sdr.rx()

    def config_gain(self, gain):
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = int(gain)
