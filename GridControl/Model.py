import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.functional as F
from RestNetBlocks import ResNetLayer, ResNetBottleNeckBlock
from Parameters import Parameters
"""
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(1, kernel_size = (3, 3), padding='same')
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x)+x)
"""

class ActorCritic(nn.Module):
    
    def __init__(self, lr, input_dims, nb_blocks):
        super(ActorCritic, self).__init__()
        self.nb_blocks = nb_blocks

        self.input_dims = input_dims #input_dims is the number of cells in the f_map input, cell_nb**2
        self.blocks = []
        self.Para = Parameters()

        """self.blocks.append(ResNetLayer(1, 1))
        self.blocks.append(ResNetLayer(1, 1, n=nb_blocks))
        
        self.sig = ResNetLayer(1, 1)
        self.mu = ResNetLayer(1, 1)

        """
        self.blocks.append(
                nn.Conv2d(self.Para.f_map_depth, 10, (3, 3), padding=1)
            )
        self.blocks.append(
                nn.Conv2d(10, 15, (3, 3), padding=1)
            )
        self.blocks.append(
                nn.Conv2d(15, 1, (3, 3), padding=1)
            )
        self.sig = nn.Conv2d(1, 1, (1, 1), padding=0)
        self.mu = nn.Conv2d(1, 1, (1, 1), padding=0)
        self.val = nn.Linear(self.input_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def forward(self, f_map):
        x = f_map.reshape([1, self.Para.f_map_depth, self.nb_blocks, self.nb_blocks])
        #print(f"x shape as input : {x.shape}")
        for ix, l in enumerate(self.blocks):
            x = l(x)
            #print(f"x shape after block {ix} : {x.shape}")
        sigma = self.sig(x)
        mu = self.mu(x)
        #print(f"shapes sigma {sigma.shape}, mu {mu.shape}, x0 {x.shape}, x1 {x.view([1, self.input_dims]).shape}")
        val = self.val(x.view([1, self.input_dims]))
        return (mu, sigma), val 



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
