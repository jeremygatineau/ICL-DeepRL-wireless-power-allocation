import pyglet
from pyglet.window import mouse
from pyglet.window import key
import numpy as np
from ToyProblem1.Parameters import Parameters
import ToyProblem1.Rendering as Rendering
from ToyProblem1.Device import Device
class Swarm:
    def __init__(self):
        self.dList = []
        self.Para = Parameters()

    def N(self):
        return len(self.dList)
    def dList_init(self, initial_conditions):
        """
        Initializes and instanciates all devices given their initial conditions; 
        initial_conditions is of the form [(x_0, y_0), (vx_0, vy_0), ..., (x_n, y_n), (vx_n, vy_n)] describing the initial parameters all devices (from 0 to n)
        """
        self.dList = []
        for dID, (pos, vel) in enumerate(initial_conditions):
            d = Device(dID, pos, vel)
            
            if True: #np.random.random() < self.Para.transmit_duty:
                d.rid = np.random.randint(0, len(initial_conditions))
                while d.rid == dID:
                    #print(f"stuck here 1 {d.rid}, {dID}")
                    d.rid = np.random.randint(0, len(initial_conditions))
            else :
                d.rid = None
            d.transmit_time = np.floor(np.random.exponential(self.Para.average_transmit_time-1))+1
            self.dList.append(d)
        
        return self.dList

    
class Environment(Swarm):
    def __init__(self, dt=0.01):
        super().__init__()
        self.initialConditions = None
        self.dt = dt
        
   
    def render(self):
        Rendering.render(self.dList, self.step, 1, [[0]])

    def step(self, action):
        old_state = self.get_state()
        for ix, device in enumerate(self.dList):
            #device.update(self.dt) #move each agent
            device.power = action[ix]*device.Pmax #apply the chosen power to each device
            device.transmit_time -= 1
            if device.transmit_time < 1 : #device finished transmitting to its assigned receiver
                if True: #np.random.random() < self.Para.transmit_duty: #to not have everyone transmitting all the time
                    device.rid = np.random.randint(0, self.N()) #new random receiver
                    while device.rid==device.id : 
                        #print(f"stuck here 2 {device.rid}, {device.id}")
                        device.rid = np.random.randint(-1, self.N())
                else :
                    device.rid = None
                device.transmit_time = np.floor(np.random.exponential(self.Para.average_transmit_time-1))+1 #for a new random transmit time 
            self.dList[ix] = device #replace the updated device from the list for safety measures
        state = self.get_state()
        episode = {"s": old_state, "r":self.objective(), "d":0, "s_":state} #construct the episode
        return episode
        
    def make(self, n_devices, init_L=None):
        if init_L is None:
            pos_L = np.random.randn(n_devices, 2)/2
            init_L = [(pos, (0,0)) for pos in pos_L]
        
        self.initialConditions = init_L
        self.dList_init(init_L)
        state = self.get_state()
        return state

        
    def get_state(self):
        return [(tx.id, list(self.dList[tx.rid].position)+list(tx.position)+[tx.power*tx.Pmax]) for tx in self.dList if tx.rid is not None]
        #return [(tx.id, list(tx.position-self.dList[tx.rid].position)+list(tx.position)+[tx.power*tx.Pmax]) for tx in self.dList if tx.rid is not None]

    def reset(self):
        assert(self.initialConditions is not None, "One has to have called Environment.make() before reseting")
        self.dList_init(self.initialConditions)
        
        for ix, d in enumerate(self.dList):
            d.power = 0
            self.dList[ix] = d
        state = self.get_state()
        return state

    def compute_scheduling(self):
        """
        returns the scheduling matrix for the current transmitters/receivers pairs using the rids in self.dlist
        """
        H = np.zeros((self.N(), self.N()))
        for dev in self.dList:
            if dev.rid is not None:
                H[dev.id][dev.rid] = 1
        
        return H

    def compute_SINR(self, D, shadowing=False, fastfading=False):
        """
        D is a matrix where each coefficient D[i,j] is the distance between device i and j
        """
        H = self.compute_scheduling()
        P = np.diag([device.power for device in self.dList]) #device power already in dBm
        D = (D+1)*self.Para.side_length/2
        #print(f"D : {D.shape}, dlist : {self.N()}")
        Tx_over_Rx = 6 + 20*np.log10(D/self.Para.Rbp)*(1+(D>self.Para.Rbp).astype(int)) # + self.Para.Lbp
        
        Path_loss = -Tx_over_Rx + P # dependence on the transmit power (in dB)
        # formerly + np.eye(s elf.N())*self.Para.Antenna_Gain 
        
        Channel_loss = np.power(10, Path_loss/10) # abs
        #print("Channel_loss before things", Channel_loss)
        if shadowing:
            Channel_loss *= np.power(10, np.random.normal(loc=0, scale=8, size=np.shape(Channel_loss))/10)
        if fastfading:
            Channel_loss *= (np.random.normal(loc=0, scale=1, size=np.shape(Channel_loss)) +\
                              np.random.normal(loc=0, scale=1, size=np.shape(Channel_loss)))/2
        #print("Channel_loss after things", Channel_loss)
        DRL = Channel_loss*np.eye(self.N()) # DirectLink Channel Loss including scheduling
        CRL = np.matmul(Channel_loss*(1-np.eye(self.N())), H) # CrossLink Channel Loss including scheduling

        SINR = DRL/(CRL+self.Para.Noise_power/self.Para.Ptx)
        #print(f"SINR : {SINR} \nDRL : {DRL}\nCRL : {CRL}\nH : {H}\nPower : {P}\nOG CRL : {Channel_loss*(1-np.eye(self.N()))}")
        
        return SINR
        
    def compute_Rates(self, SINR): 
        """
        SINR is a matrix where each coefficient SINR[i,j] represents the cross-SINR between device i and j
        """
        return np.log2(1+SINR/self.Para.SNRgap)#*self.Para.Bandwith

    def apply_power(self, power_vector):
        assert(len(power_vector==self.N()), "power vector not of invalid length")
        for i in range(self.N()):
            self.dList[i].power = power_vector[i] 
        
        return self.objective()

    def objective(self):
        D = np.zeros([len(self.dList), len(self.dList)])
        for j, dj in enumerate(self.dList):
            for i, di in enumerate(self.dList[j:]):
                D[j, j+i] = np.linalg.norm(dj.position-di.position)
                D[j+i, j] = D[j, j+i]

        SINR = self.compute_SINR(D)
        rates = self.compute_Rates(SINR)
        sum_rate = np.sum(rates.flatten())

        #Clip SINR to the clipping thershold defined in parameters:
        for ir, row in enumerate(SINR):
            for ix, sinr in enumerate(row):
                if 10*np.log10(sinr)>self.Para.SINRClip:
                    SINR[ir, ix] = self.Para.SINRClip
        
        #print(f"D {D} \nSINR dB {np.round(10*np.log10(SINR))} \nRATES {rates} \nsum_rate {sum_rate}")

        return sum_rate#np.std([ra for ra in rates.flatten() if ra>0]))
        """
        H = compute_gains()
        p = lambda j : self.dList[j].power
        rate = lambda i : np.log(1+ (H[i, i]**2*p[i] /sum([H[i, j]**2 * p[j] for j in range(len(self.dList))])))
        return sum([rate[i] for i in range(len(self.dList))])"""
        

        

