from Model import ActorCritic

import torch
from torch import nn
from torch.utils import data
import pandas as pd
from pandas import DataFrame
import time as t
import numpy as np
import torch.nn.functional as F
from Parameters import Parameters

class Agent:

    def __init__(self, cell_nb,lr=4e-3, nb_blocks=5,  gamma=0.99):
        self.cell_nb = cell_nb
        self.gamma = gamma
        self.ActorCritic = ActorCritic(lr, cell_nb**2, nb_blocks)

        self.device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"model pushed to {self.device}")
            self.ActorCritic = self.ActorCritic.cuda()
            
        self.log_probs = None
        self.Para = Parameters()

    def choose_action(self, state): #here state is simply the current f_map
        
        state_tensor = torch.tensor([state]).float().to(self.device)

        (mu, sigma), _ = self.ActorCritic.forward(state_tensor)
        mu = mu.reshape([self.cell_nb, self.cell_nb])
        sigma = sigma.reshape([self.cell_nb, self.cell_nb])
        actions = np.zeros([self.cell_nb, self.cell_nb])
        log_probs = np.zeros_like(actions)
        for ir, (mu_r, sig_r) in enumerate(zip(mu, sigma)):
            for ic, (mu_c, sig_c) in enumerate(zip(mu_r, sig_r)):
                #mu_c and sig_c are the mu and sigma parameter for the gaussian distribution of the current cell
                sig_c = torch.exp(sig_c)
                dist = torch.distributions.Normal(mu_c, sig_c)
                action = dist.sample()
                log_prob = dist.log_prob(action).to(self.device)
                actions[ir, ic] = F.sigmoid(action) #bound the normalized transmit power between 0 and 1
                log_probs[ir, ic] = log_prob #for later, to calculate the actor loss
        self.log_probs = log_probs
        return actions


    def learn(self, episode):
        
        self.ActorCritic.optimizer.zero_grad()

        #s is the state, in the most simple case it is the f_map
        f_map = torch.tensor(episode["s"], requires_grad=True).float().to(self.device) #current f_map
        r = torch.tensor(episode["r"], requires_grad=True).float().to(self.device) #the embeded objective function (sum-rate, capcity, SINR...)
        d = torch.tensor(episode["d"]).bool().to(self.device) #done, not really necessary
        f_map_ = torch.tensor(episode["s_"], requires_grad=True).float().to(self.device) #new f_map
        lg_p = torch.tensor(self.log_probs, requires_grad=True).float().to(self.device) #log_probs as given by choose_action
        
        f_map = f_map.reshape([1, self.Para.f_map_depth,self.cell_nb, self.cell_nb])
        f_map_ = f_map_.reshape([1, self.Para.f_map_depth,self.cell_nb, self.cell_nb])
        #get critic values for current and next state
        _, val = self.ActorCritic.forward(f_map) 
        _, val_ = self.ActorCritic.forward(f_map_)

        #set the values for the next state to 0 if done
        val_[d] = 0.0

        #compute the delta
        
        delta = r + self.gamma * val_.item() - val.item()
        
        #print(f"r {r} shape {r.shape}")
        actor_loss = -torch.mean(lg_p.flatten()*delta)
        critic_loss = delta**2/100

        (actor_loss + critic_loss).backward()
        self.ActorCritic.optimizer.step()

        return actor_loss.item(), critic_loss.item()
    



    