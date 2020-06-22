import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torch.autograd import Variable



class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(1, kernel_size = (3, 3), padding='same')
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x)+x)
        
class Model(nn.Module):
    def __init__(self, cell_nb, nb_blocks):
        super(Model, self).__init__()
        self.nb_blocks = nb_blocks

        self.blocks = []
        for _ in range(nb_blocks):
            self.blocks.append(ConvBlock())
        
    def forward(self, f_map):
        x = f_map
        for l in self.blocks:
            x = l(x) + x
        
        return x
    