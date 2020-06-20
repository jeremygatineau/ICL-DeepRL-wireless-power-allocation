import pyglet
from pyglet.window import mouse
from pyglet.window import key
import numpy as np
from Device import Device
import rendering
class Swarm:
    def __init__(self, cell_nb=5):
        self.dList = []
        self.cell_nb = cell_nb
        pass
    
    def discretize(self):
        """
        Creates the frequency map from the list of devices and the number of cells.
        """
        assert(self.dList != [], "Devices not initialized, call dList_init before creating the frequency map.")
        
        f_map = np.zeros((self.cell_nb, self.cell_nb))

        for dev in self.dList:
            x, y = dev.position
            cx = np.floor((x+1)*self.cell_nb/2) + 1
            cy = np.floor((y+1)*self.cell_nb/2) + 1
            
            assert(cx>1 and cx<=self.cell_nb and cy>1 and cy<=self.cell_nb, f"Device {dev.ID} out of bound (position tuple {(dev.position[0], dev.position[0])}).")
            
            f_map[cy][cx] += 1
        
        return f_map


    def dList_init(self, initial_conditions):
        """
        Initializes and instanciates all devices given their initial conditions; 
        initial_conditions is of the form [(x_0, y_0), (vx_0, vy_0), ..., (x_n, y_n), (vx_n, vy_n)] describing the initial parameters all devices (from 0 to n)
        """

        for dID, (pos, vel) in enumerate(initial_conditions):
            self.dList.append(Device(dID, pos, vel))
        
        return self.dList
    def compute_gains(self):
        gain_mat = np.zeros(len(self.dList), len(self.dList))
        for d1 in self.dList:
            for d2 in self.dList:
                if d1.id == d2.id:
                    gain_mat[d1.id, d1.id] = 1
                else :
                    gain_mat[d1.id, d2.id] = 1/np.linalg.norm(d1.position - d2.position)**2
        return gain_mat
    def objective(self):
        H = compute_gains()
        p = lambda j : self.dList[j].power
        rate = lambda i : np.log(1+ (H[i, i]**2*p[i] /sum([H[i, j]**2 * p[j] for j in range(len(self.dList))])))
        return sum([rate[i] for i in range(len(self.dList))])
    
    def render(self):

        def update(dt):
            for device in self.dList:
                device.update(dt)
        
        rendering.render(self.dList, update)

