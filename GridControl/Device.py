import numpy as np

class Device:
    def __init__(self, ID, initial_position = (0,0), initial_velocity = (0,0)):
        self.position = np.array(initial_position)
        self.power = 0
        self.velocity = np.array(initial_velocity)
        self.id = ID
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
        d = np.linalg.norm(self.position)
        v = 0.1/d**2
        mat = np.array([
            [np.cos(dt*v), -np.sin(dt*v)],
            [np.sin(dt*v), np.cos(dt*v)]
        ])
        #self.position = dt*self.velocity + self.position
        self.position = np.dot(mat, self.position)

        self.position[0] =  max(-0.99, min(0.99, self.position[0]))
        self.position[1] = max(-1, min(0.999, self.position[1]))
        return self.position

    def getPowerFroPolicy(self, policy):
        cell_nb = policy.shape[0]
        x = (self.position[0]+1)/2
        x = np.floor(cell_nb*x)
        y = (self.position[1]+1)/2
        y = np.floor(cell_nb*y)
        power = policy[int(x), int(y)]
        return power