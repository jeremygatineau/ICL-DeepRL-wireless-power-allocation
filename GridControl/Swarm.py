import numpy as np
from numpy.random import random
from Device import Device
class Swarm:
    def __init__(self, device_nb):
        self.deviceList = None
        pass
    
    def initDevices(self, device_nb, positions=None):
        """
        Instanciates the list of devices in the system.
        
        Positions represents the list of positions for each device, if not given, defaults to random
        """
        assert((len(positions)==device_nb) or (positions is None), "List of device positions should be of the same lengths as the number of devices.")
        
        if positions is None:
            positions = random((device_nb, 2))
        self.deviceList = []
        for i in range(device_nb):
            self.deviceList.append(Device(i, positions[i]))

        return self.deviceList

    def getPositions(self):
        assert(self.deviceList != None, "Device list needs to be initialized.")

        return [dev.position for dev in self.deviceList]
