


class Agent:

    def __init__(self, S):
        self.swarm = S
        self.model = ...

    def plan(self, f_map):
        p = self.model(f_map)
        return p

    def evaluate(self, f_map):
        sum_rate = self.swarm.compute_gains()