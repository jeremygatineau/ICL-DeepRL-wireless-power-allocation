from ToyProblem1.Model import ActorCritic

import torch
from torch import nn
from torch.utils import data
import pandas as pd
from pandas import DataFrame
import time as t
import numpy as np
import torch.nn.functional as F
from ToyProblem1.Parameters import Parameters
from ToyProblem1.Agent import ActorCritic
class Agent:

    def __init__(self, nb_devices, lr=4e-3,  gamma=0.99):
        self.gamma = gamma
        self.ActorCritic = ActorCritic(lr, nb_devices*5, nb_devices, 20)
        self.nb_devices = nb_devices
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            print(f"model pushed to {self.device}")
            self.ActorCritic = self.ActorCritic.cuda()
            self.ActorCritic.optimizer.cuda()
            
        self.log_probs = None
        self.Para = Parameters()

    def choose_action(self, state): #here state is simply the current f_map
        #print(f"state in choose action {state}")
        state_tensor = torch.tensor(np.array([list(dist)+ list(pos) + [power] for dist, pos, power in state]).flatten()).float().to(self.device)
        #print(f"state tensor {state_tensor}")
        (mu, sigma), _ = self.ActorCritic.forward(state_tensor)
        mu = mu.reshape(self.nb_devices)
        sigma = sigma.reshape(self.nb_devices)
        actions = np.zeros(self.nb_devices)
        log_probs = np.zeros_like(actions)
        
        for ix, (mu_, sig_) in enumerate(zip(mu, sigma)):
            #mu_c and sig_c are the mu and sigma parameter for the gaussian distribution of the current cell
            sig_ = torch.exp(sig_)
            dist = torch.distributions.Normal(mu_, sig_)
            action = dist.sample()
            log_prob = dist.log_prob(action).to(self.device)
            actions[ix] = F.sigmoid(action) #bound the normalized transmit power between 0 and 1
            log_probs[ix] = log_prob #for later, to calculate the actor loss

        self.log_probs = log_probs
        return actions


    def learn(self, episode):
        
        self.ActorCritic.optimizer.zero_grad()
        #print(episode)
        #s is the state, in the most simple case it is the f_map
        print(f"state in learn {episode['s']}")
        s = np.array([(dist, pos, power) for _, dist, pos, power in episode["s"]]).flatten() #NEED TO INCLUDE THE TRANSMIT POWER FOR THE CRITIC TO GUESS THE SINR HOW COULD YOU NOT SEE THAT LIKE 1 MONTH AGO YOU STUPID CUNT
        s = torch.tensor(s, requires_grad=True).float().to(self.device) #current state
        r = torch.tensor(episode["r"], requires_grad=True).float().to(self.device) #the embeded objective function (sum-rate, capcity, SINR...)
        d = torch.tensor(episode["d"]).bool().to(self.device) #done, not really necessary
        s_ = list(np.array([(dist, pos, power) for _, dist, pos, power in episode["s_"]]).flatten())
        s_ = torch.tensor(s_, requires_grad=True).float().to(self.device) #new state
        lg_p = torch.tensor(self.log_probs, requires_grad=True).float().to(self.device) #log_probs as given by choose_action
        

        #get critic values for current and next state
        #_, val = self.ActorCritic.forward(s) 
        _, val_ = self.ActorCritic.forward(s_)

        #set the values for the next state to 0 if done
        #val_[d] = 0.0

        #compute the delta
        
        delta = r  - val_ #+ self.gamma * val_ - val
        #print(f"delta {delta} \nval {val} \nval_ {val_}\n\n")
        #print(f"r {r} f_map {f_map} lg_p {lg_p}")
        actor_loss = -torch.mean(lg_p.flatten()*delta)
        critic_loss = delta**2
        print(f"--actual reward {np.round(r.detach().numpy(), 2)} \n--predicted value for state {np.round(val_.detach().numpy(), 2)}\n--difference of {np.round(delta.detach().numpy(), 4)}")
        critic_loss.backward()
        #(actor_loss + critic_loss).backward()
        self.ActorCritic.optimizer.step()

        return actor_loss.item(), critic_loss.item()
    



    