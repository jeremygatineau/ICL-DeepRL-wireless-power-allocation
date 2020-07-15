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
        self.Para = Parameters()
        
    def discretize(self):
        """
        Creates the frequency map from the list of devices and the number of cells.
        """
        assert(self.dList != [], "Devices not initialized, call dList_init before creating the frequency map.")
        
        f_map = [[[]for k in range(self.cell_nb)]for k in range(self.cell_nb)]  #np.zeros((self.cell_nb, self.cell_nb, 1))
        
        for dev in self.dList:
            x, y = dev.position

            x = max(-1, min(0.999, x))
            y = max(-1, min(0.999, y))

            cx = int(np.floor((x+1)*self.cell_nb/2))
            cy = int(np.floor((y+1)*self.cell_nb/2))

            #print(f"dev : {dev.position}\ndev_r : {self.dList[dev.rid].position}")
            if dev.rid is not None:
                dist = np.linalg.norm(dev.position - self.dList[dev.rid].position)
                f_map[cy][cx].append(dist)

        for i in range(len(f_map)):
            for j in range(len(f_map)):
                f_map[i][j] = np.pad(sorted(f_map[i][j]), (0, max(0, self.Para.f_map_depth-len(f_map[i][j]))))[:self.Para.f_map_depth] #pad and trim to the right size
        
        self.f_map = f_map
        return f_map

    def N(self):
        return len(self.dList)
    def dList_init(self, initial_conditions):
        """
        Initializes and instanciates all devices given their initial conditions; 
        initial_conditions is of the form [(x_0, y_0), (vx_0, vy_0), ..., (x_n, y_n), (vx_n, vy_n)] describing the initial parameters all devices (from 0 to n)
        """
        self.dList = []
        for dID, (pos, vel) in enumerate(initial_conditions):
            self.dList.append(Device(dID, pos, vel))
            self.dList[dID].rid = np.random.randint(0, len(initial_conditions))
            while self.dList[dID].rid == dID:
                print(f"stuck here 1 {self.dList[dID].rid}, {dID}")
                self.dList[dID].rid = np.random.randint(0, len(initial_conditions))
            self.dList[dID].transmit_time = np.floor(np.random.exponential(self.Para.average_transmit_time-1))+1
        
        return self.dList

    
class Environment(Swarm):
    def __init__(self, cell_nb=5, dt=0.01):
        super().__init__(cell_nb=cell_nb)
        self.initialConditions = None
        self.dt = dt
        
   
    def render(self):
        Rendering.render(self.dList, self.step, self.cell_nb, self.f_map, self)

    def step(self, action):
        old_state = self.f_map
        for ix, device in enumerate(self.dList):
            device.update(self.dt) #move each agent
            device.power = device.getPowerFromPolicy(action) #apply the chosen power to each device
            device.transmit_time -= 1
            if device.transmit_time < 1 : #device finished transmitting to its assigned receiver
                device.rid = np.random.randint(0, self.N()) #new random receiver
                while device.rid==device.id : 
                    print(f"stuck here 2 {device.rid}, {device.id}")
                    device.rid = np.random.randint(0, self.N())
                device.transmit_time = np.floor(np.random.exponential(self.Para.average_transmit_time-1))+1 #for a new random transmit time 
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

    def compute_scheduling(self):
        """
        returns the scheduling matrix for the current transmitters/receivers pairs using the rids in self.dlist
        """
        H = np.zeros((self.N(), self.N()))
        for dev in self.dList:
            H[dev.id][dev.rid] = 1
        
        return H

    def compute_SINR(self, D, shadowing=False, fastfading = False):
        """
        D is a matrix where each coefficient D[i,j] is the distance between device i and j
        """
        H = self.compute_scheduling()
        P = np.diag([device.power for device in self.dList])
        D = (D+1)*self.Para.side_length/2
        #print(f"D : {D.shape}, dlist : {self.N()}")
        Tx_over_Rx = 6 + 20*np.log10(D/self.Para.Rbp)*(1+(D>self.Para.Rbp).astype(int)) # + self.Para.Lbp
        
        Path_loss = -Tx_over_Rx + P # dependence on the transmit power (in dB)
        # formerly + np.eye(self.N())*self.Para.Antenna_Gain 
        
        Channel_loss = np.power(10, Path_loss/10) # abs
        #print("Channel_loss before things", Channel_loss)
        if shadowing:
            Channel_loss *= np.power(10, np.random.normal(loc=0, scale=8, size=np.shape(Channel_loss))/10)
        if fastfading:
            Channel_loss *= (np.random.normal(loc=0, scale=1, size=np.shape(Channel_loss)) +\
                              np.random.normal(loc=0, scale=1, size=np.shape(Channel_loss)))/2
        print("Channel_loss after things", Channel_loss)
        DRL = Channel_loss*np.eye(self.N()) # DirectLink Channel Loss including scheduling
        CRL = np.matmul(Channel_loss*(1-np.eye(self.N())), H) # CrossLink Channel Loss including scheduling

        SINR = DRL/(CRL+self.Para.Noise_power/self.Para.Ptx)
        print(f"SINR : {SINR} \nDRL : {DRL}\nCRL : {CRL}\nH : {H}\nPower : {P}\nOG CRL : {Channel_loss*(1-np.eye(self.N()))}")
        a = 1
        a.append(1)
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
        return np.sum(self.compute_Rates(SINR).flatten())
        """
        H = compute_gains()
        p = lambda j : self.dList[j].power
        rate = lambda i : np.log(1+ (H[i, i]**2*p[i] /sum([H[i, j]**2 * p[j] for j in range(len(self.dList))])))
        return sum([rate[i] for i in range(len(self.dList))])"""
        

        

