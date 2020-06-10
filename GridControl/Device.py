import numpy as np

class Device:
    def __init__(self, initial_position = (0,0)):
        self.position = np.array(initial_position)
        self.transmit_power = 0
        self.velocity = np.array((2,2))
    
    def TP_Policy(self, state):
        """
        Updates and returns the device's transmit power.
        Inputs:
            state  > the current state of the environment from the agent's perspective, the localized context for the agent.
        """
        pass
    
    def update(self, dt):
        """
        Updates and retruns the agent's position for time dt.
        """
        v = 20
        mat = np.array([
            [np.cos(dt*v), -np.sin(dt*v)],
            [np.sin(dt*v), np.cos(dt*v)]
        ])
        #self.position = dt*self.velocity + self.position
        self.position = np.dot(mat, self.position)

        return self.position
