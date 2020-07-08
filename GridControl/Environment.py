import pyglet
from pyglet.window import mouse
from pyglet.window import key
import numpy as np
from Device import Device
import Rendering as Rendering
from Parameters import Parameters


class Swarm:
    def __init__(self, cell_nb=5):
        self.dList = []
        self.cell_nb = cell_nb
        self.f_map = np.zeros((cell_nb, cell_nb))
        
    def discretize(self):
        """
        Creates the frequency map from the list of devices and the number of cells.
        """
        assert(self.dList != [], "Devices not initialized, call dList_init before creating the frequency map.")
        
        f_map = np.zeros((self.cell_nb, self.cell_nb))

        for dev in self.dList:
            x, y = dev.position

            x = max(-1, min(0.999, x))
            y = max(-1, min(0.999, y))

            cx = int(np.floor((x+1)*self.cell_nb/2))
            cy = int(np.floor((y+1)*self.cell_nb/2))

            #print(f"cx, cy {cx, cy}; x, y {x, y}")
            assert(cx>1 and cx<=self.cell_nb and cy>1 and cy<=self.cell_nb, f"Device {dev.id} out of bound (position tuple {(dev.position[0], dev.position[0])}).")
            
            f_map[cy][cx] += 1
        self.f_map = f_map
        return f_map
    def N(self):
        return len(self.dList)
    def dList_init(self, initial_conditions):
        """
        Initializes and instanciates all devices given their initial conditions; 
        initial_conditions is of the form [(x_0, y_0), (vx_0, vy_0), ..., (x_n, y_n), (vx_n, vy_n)] describing the initial parameters all devices (from 0 to n)
        """

        for dID, (pos, vel) in enumerate(initial_conditions):
            self.dList.append(Device(dID, pos, vel))
        
        return self.dList

    
class Environment(Swarm):
    def __init__(self, cell_nb=5, dt=0.01):
        super().__init__(cell_nb=cell_nb)
        self.initialConditions = None
        self.dt = dt
        self.Para = Parameters()
   
    def render(self):
        Rendering.render(self.dList, self.step, self.cell_nb, self.f_map, self)

    def step(self, action):
        old_state = self.f_map
        for ix, device in enumerate(self.dList):
            device.update(self.dt) #move each agent
            device.power = device.getPowerFromPolicy(action) #apply the chosen power to each device
            self.dList[ix] = device #replace the updated device from the list for safety measures
        self.discretize() #rebuild the f_map

        episode = {"s": old_state, "r":self.objective(), "d":0, "s_":self.f_map} #construct the episode
        return episode
        
    def make(self, n_devices, init_L=None):
        if init_L is None:
            pos_L = np.random.randn(n_devices, 2)/2
            init_L = [(pos, (0,0)) for pos in pos_L]
        
        self.initialConditions = init_L
        self.dList_init(init_L)

    def reset(self):
        assert(self.initialConditions is not None, "One has to have called Environment.make() before reseting")
        self.dList_init(self.initialConditions)
        self.discretize()
        
        for ix, d in enumerate(self.dList):
            d.power = 0
            self.dList[ix] = d

        return self.f_map

    def compute_SINR(self, D, shadowing=True, fastfading = True):
        """
        D is a matrix where each coefficient D[i,j] is the distance between device i and j
        """
        P = np.diag([device.power for device in self.dList])

        Tx_over_Rx = self.Para.Lbp + 6 + 20*np.log10(D/self.Para.Rbp)*(1+(D>self.Para.Rbp).astype(int))

        Path_loss = -Tx_over_Rx + P # dependence on the transmit power (in dB)
        # formerly + np.eye(self.N())*self.Para.Antenna_Gain 
        
        Channel_loss = np.power(10, Path_loss/10) # abs
        if shadowing:
            Channel_loss *= np.power(10, np.random.normal(loc=0, scale=8, size=np.shape(Channel_loss))/10)
        if fastfading:
            Channel_loss *= (np.random.normal(loc=0, scale=1, size=np.shape(Channel_loss)) +\
                              np.random.normal(loc=0, scale=1, size=np.shape(Channel_loss)))/2

        DRL = Channel_loss*np.eye(self.N()) # DirectLink Channel Loss
        CRL = Channel_loss*(1-np.eye(self.N())) # CrossLink Channel Loss

        SINR = DRL/(CRL+self.Para.Noise_power/self.Para.Ptx)
        return SINR
        
    def compute_Rates(self, SINR):
        """
        SINR is a matrix where each coefficient SINR[i,j] represents the cross-SINR between device i and j
        """
        return self.Para.Bandwith*np.log2(1+SINR/self.Para.SNRgap)
        
    def objective(self):
        D = np.zeros([len(self.dList), len(self.dList)])
        for j, dj in enumerate(self.dList):
            for i, di in enumerate(self.dList[j:]):
                D[j, j+i] = np.linalg.norm(dj.position-di.position)
                D[j+i, j] = D[j, j+i]

        SINR = self.compute_SINR(D)
        return self.compute_Rates(SINR)
        """
        H = compute_gains()
        p = lambda j : self.dList[j].power
        rate = lambda i : np.log(1+ (H[i, i]**2*p[i] /sum([H[i, j]**2 * p[j] for j in range(len(self.dList))])))
        return sum([rate[i] for i in range(len(self.dList))])"""
        

        

