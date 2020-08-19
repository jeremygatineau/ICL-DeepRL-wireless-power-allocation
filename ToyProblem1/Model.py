import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from ToyProblem1.Parameters import Parameters

class Critic(nn.Module):
    
    def __init__(self, lr, input_size, output_size, hidden_size):
        super(Critic, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lr = lr

        self.Para = Parameters()
    
        
        self.body = nn.ModuleList(
            [nn.Linear(input_size, hidden_size),
            nn.ReLU()] +
            [nn.Linear(hidden_size, hidden_size),
            nn.Tanh()]*7
        )

        self.val = nn.Linear(hidden_size, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        

    def forward(self, x):
        #print(f"x shape as input : {x.shape}")
        for ix, b in enumerate(self.body):
            if ix==0:
                x = b(x)
            else :
                x =b(x) #+ x

        
        val = self.val(x)
        #print(f"shapes sigma {sigma.shape}, mu {mu.shape}, x {x.shape}, val {val.shape}")
        
        #print(f"mu {mu} \nsigma {sigma}")
        return val 

class Actor(nn.Module):
    
    def __init__(self, lr, input_size, output_size, hidden_size):
        super(Actor, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lr = lr

        self.Para = Parameters()
        #print(f"input_size {input_size}")
        self.body = nn.ModuleList(
            [nn.Linear(input_size, hidden_size),
            nn.ReLU()] +
            [nn.Linear(hidden_size, hidden_size),
            nn.Tanh()]*7
        )

        self.sig = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Sigmoid())
        self.mu = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Sigmoid())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        

    def forward(self, x):
        #print(f"x shape as input : {x.shape}")
        for ix, b in enumerate(self.body):
            if ix==0:
                x = b(x)
            else :
                x =b(x) + x
        sigma = self.sig(x)
        mu = self.mu(x)

        #print(f"shapes sigma {sigma.shape}, mu {mu.shape}, x {x.shape}, val {val.shape}")
        
        #print(f"mu {mu} \nsigma {sigma}")
        return mu, sigma



class Dataset(Dataset):
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
