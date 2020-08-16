from queue import LifoQueue
import numpy as np



class State:
    def __init__(self):
        local_info = {"p(t-1)" : None, "w(t)" : None, "C(t-1)": None, "g_ii(t)": None, "g_ii(t-1)": None, "sum_j!=i g_ji(t)pj(t-1)+sig": None, "sum_j!=i g_ji(t-1)pj(t-2)+sig": None}
        interfering_neigh_info = {}
        interfered_neigh_info = {}
²


    


    
class Device:
    def __init__(self, tid, Pmax, buffer_size=0, initial_conditions = {"pos" : None, "vel" : None}):
        self.pos = np.array(initial_conditions["pos"]) is initial_conditions["pos"] is not None
        self.vel = np.array(initial_conditions["vel"]) is initial_conditions["vel"] is not None

        self.p = 0
        self.Pmax = Pmax

        self.id = tid

        self.rid = None

        self.buffer = LifoQueue(buffer_size)

        self.current_frame = None

    def transmit(self, n, time_step, rate):
        if current_frame is not None:
            currrent_frame["len"] -= time_step*rate
            cf = current_frame
            if current_frame["len"]<0 : 
                current_frame = None            
        else : 
            cf = self.buffer.get()
            cf["len"] = time_step*rate

            if cf["len"] > 0 :
                self.current_frame = cf
        return cf
    
    def __repr__(self):
        return f"(device {self.id}, pos {self.position}, rec {self.rid})"

d = Device(6, 0)
print(d)