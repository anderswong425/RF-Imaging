# RF Imaging with PlutoSDR

This repository contains the codebase for the RF Imaging System, which utilizes PlutoSDR for real-time image reconstruction. The system is capable of capturing an environment using received RF signals and reconstructing images based on the signal propagation.

## Software Setup
1. Clone the repository:
    ```
    git clone https://github.com/anderswong425/RF-Imaging.git
    ```

2. Install PlutoSRD driver:
    ```
    . PlutoSRD_driver_installation.sh
    ```

3. Update PlutoSDR firmware:
    1. Download the plutosdr-fw-vX.XX.zip from https://github.com/analogdevicesinc/plutosdr-fw/releases/latest
    2. Unzip all files onto the mass storage device that you want to update
    3. Eject the device and LED1 will start blinking rapidly
    
        (Once the device is done programming, it will appear as a mass storage device again.)
    4. Edit [NETWORK] in config.txt, for example: 
        ```
        [NETWORK]
        hostname = pluto20
        ipaddr = 192.168.21.1
        ipaddr_host = 192.168.21.10
        netmask = 255.255.255.0
        ```
    5. Eject it again to make changes 

4. Install Python dependencies:
    ```
    pip3 install -r requirements.txt
    ```

## Hardware Setup
- To create the imaging domain, the PlutoSDR boards are placed in an evenly spaced and anticlockwise configuration, as shown in the diagram below:
![Alt text](/result/transceiver_config.png)

## Script Flow
The RF Imaging System script follows a designed flow to efficiently capture RF signals and reconstruct images in real-time. Here is an overview of the main steps:

1. **Parameter Initialization**: The system automatically calculates any necessary parameters for the reconstruction process based on the user-defined parameters in the main.py file.

2. **PlutoSDR Initialization**: The system initializes the PlutoSDR boards based on the specified device indices and sets relevant parameters such as frequency, number of samples, gain, and more.

3. **Data Collection**: The PlutoSDR boards take turns transmitting signals, while the remaining boards receive these signals. The first collected dataset (**Pinc**) serves as the reference for background subtraction to eliminate noise and artifacts.

4. **Reconstruction and Visualization**: The collected dataset (**Ptot**) and the reference dataset are passed to the reconstruction and regularization algorithms in the real_time_visualization function. This process continuously updates the reconstruction and visualizes the real-time images.

## Running the code
1. Adjust the parameters(frequency, resolution, denoising weight, etc.) by editing main.py

2. Run the real-time system:
    ```
    python3 main.py
    ```
