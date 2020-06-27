import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.functional as F


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(1, kernel_size = (3, 3), padding='same')
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x)+x)
        

class ActorCritic(nn.Module):
    
    def __init__(self, lr, input_dims, nb_blocks):
        super(ActorCritic, self).__init__()
        self.nb_blocks = nb_blocks

        self.input_dims = input_dims #input_dims is the number of cells in the f_map input, cell_nb**2
        self.blocks = []
        for _ in range(nb_blocks):
            self.blocks.append(ConvBlock())
        
        self.sig = ConvBlock()
        self.mu = ConvBlock()
        self.val = nn.Linear(self.input_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, f_map):
        x = f_map
        for l in self.blocks:
            x = F.ReLU(l(x) + x)
        sigma = self.sig(x)
        mu = self.mu(x)
        val = self.val(x.view([1, self.input_dims, 1]))
        return (mu, sigma), val 



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