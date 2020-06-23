from model import Model

import torch
from torch import nn
from torch.utils import data
import pandas as pd
from pandas import DataFrame
import time as t
import numpy as np


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, labels):
        'Initialization'

        self.labels = labels
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        X = self.data.iloc[[index]].values
        y = self.labels.iloc[[index]].values
        return X, y

class Agent:

    def __init__(self, S, epochs=10):
        self.swarm = S
        self.model = Model(S.cell_nb, 5)
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"

    def plan(self):
        p = self.model(self.S.f_map)
        return p

    def evaluate(self):
        sum_rate = self.swarm.compute_gains()
        ...
    def online_training(self, update_timestep):
        
        for k in range(self.epochs):
            pos_buf = Dataframe()     
            for t in range(update_timestep):
                f_map = self.S.discretize() 
                d = Dataframe({"f_map" : f_map, "gains" : self.compute_gains(), "t": t})
                pos_buf = pd.concat(pos_buf, d)
                self.S.step()

            ds = Dataset(pos_buf[["f_map"]], pos_buf[["gains"]])

            for _ in range(self.t_epochs):
                self.train1Epoch(ds)

    def train1Epoch(self, train_Dataset):  # returns the losses and the time it took to train
            t_st = t.time()
            train_gen = data.DataLoader(train_Dataset, batch_size=self.batch_size, shuffle=True)
            
            optimizer = torch.optim.Adam(
                self.Pol.parameters(), lr=self.lr)

            listoss = []
            for x, y in train_gen:
                x = x.to(device)
                y = y.to(device)
                output = self.model.forward(x.float())
                loss = self.compute_loss(output, y.float())
                listoss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            return listoss, t.time() - t_st
    
    def compute_loss(self, gains, policy):

        ps = np.array([d.getPowerFromPolicy(policy) for d in self.S.dList()])
        H = gains
        rate = [np.log(1+ (H[i, i]**2*p_ /sum([H[i, j]**2 * p for j, p in enumerate(ps)]))) for i, p_ in enumerate(ps)]
        return -np.sum(rate)

    def objective(self):
        return 0
            



    