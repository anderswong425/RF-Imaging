# RF Imaging with PlutoSDR

This repository contains the codebase for the RF Imaging System, which utilizes PlutoSDR for real-time image reconstruction. The system is capable of capturing an environment using received RF signals and reconstructing images based on the signal propagation.

## Setup
1. Clone the repository:
```
git clone https://github.com/anderswong425/RF-Imaging.git
```

2. Install PludoSRD driver:
```
. PludoSRD_driver_installation.sh
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

## Running the code
1. Adjust the parameters(frequency, resolution, denoising weight, ...) by editing main.py

2. Run the real-time system:
```
python3 main.py
```