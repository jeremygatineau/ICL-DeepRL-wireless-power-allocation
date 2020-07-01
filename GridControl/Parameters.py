import numpy as np

class Parameters:
    def __init__(self):
        self.Tx_height = 1.5 # m
        self.Rx_height = 1.5 # m

        self.Bandwith = 5e6 # Hz
        self.Carrier_frequency = 2.4e9 # Hz
        self.Wavelength = 299792458/self.Carrier_frequency # m

        self.Rbp = 4*self.Tx_height*self.Rx_height/self.Wavelength
        self.Lbp = 120*np.log10(self.Wavelength**2/(8*np.pi*self.Tx_height*self.Rx_height))

        self.Antenna_Gain = 2.5

        self.Ptx = 40 # mdB

        self.SNRgapdB = 6 # dB
        self.SNRgap = 10**(self.SNRgapdB/10)

        self.Noise_power_density = -169 # mdB
        self.Noise_power = self.Bandwith*10**((self.Noise_power_density-30)/10)




