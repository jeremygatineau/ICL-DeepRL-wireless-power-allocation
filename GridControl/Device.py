import numpy as np

class Device:
    def __init__(self, id, initial_position = np.array((0,0)), desactivate=0.5, activate=0.5, mobility=True):
        self.position = initial_position
        self.transmit_power = 0
        self.activity = {"10" : desactivate, "01": activate}
        self.mobility = mobility
    def TP_Policy(self, state):
        """
        Updates and returns the device's transmit power.
        Inputs:
            state  > the current state of the environment from the agent's perspective, the localized context for the agent.
        """
        pass
    
    def move(self, delta):
        """
        Updates and retruns the agent's position according to the given delta.
        Inputs:
            delta (2D np array) > the positional delta to the new position
        """
        self.position += delta
        return self.position


    