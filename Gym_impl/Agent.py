from Gym_impl.Model import Actor, Critic

import torch
from torch import nn
from torch.utils import data
import pandas as pd
from pandas import DataFrame
import time as t
import numpy as np
import torch.nn.functional as F
from Gym_impl.Parameters import Parameters
class Agent:

    def __init__(self, nb_devices, lr=4e-3,  gamma=0.9):
        self.gamma = gamma
        self.Actor = Actor(lr, nb_devices*4, nb_devices, 50)
        self.Critic = Critic(lr, nb_devices*5, nb_devices, 10)
        self.nb_devices = nb_devices
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            print(f"model pushed to {self.device}")
            self.Critic = self.Critic.cuda()
            self.Actor = self.Actor.cuda()
            self.Critic.optimizer.cuda()
            self.Actor.optimizer.cuda()
            
        self.log_probs = None
        self.Para = Parameters()

    def choose_action(self, state): #here state is simply the current f_map
        #print(f"state in choose action {state}")
        state_tensor = torch.tensor(np.array(state).flatten()).float().to(self.device)
        #print(f"state tensor {state_tensor}")
        mu, sigma= self.Actor.forward(state_tensor.view([1, self.nb_devices*4]))
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
    """
    def get_minibatch(self, state, batch_size):
        state_tensor = torch.tensor(np.array(state).flatten()).float().to(self.device)
        print(f"state {state}")
        mu, sigma= self.Actor.forward(state_tensor.view([1, self.nb_devices*4]))
        mu = mu.reshape(self.nb_devices)
        sigma = sigma.reshape(self.nb_devices)
        actions = np.zeros([self.nb_devices, batch_size])
        log_probs = np.zeros_like(actions)
        
        for ix, (mu_, sig_) in enumerate(zip(mu, sigma)):
            #mu_c and sig_c are the mu and sigma parameter for the gaussian distribution of the current cell
            sig_ = torch.exp(sig_)
            dist = torch.distributions.Normal(mu_, sig_)
            actions[ix] = F.sigmoid(dist.sample([1, batch_size]))
        actions = actions.reshape([batch_size, self.nb_devices])
        print(actions.shape)
        print(actions)
        critic_state = [[state[i] + action for action in actions] #strap the actions at the end of the state to then feed into the critc net
        return critic_state"""

    def learn_critic(self, minibatch, labels):
        x = torch.tensor(minibatch, requires_grad=True).float().to(self.device)
        y = torch.tensor(labels, requires_grad=True).float().to(self.device)
        x = x.reshape([len(minibatch), self.nb_devices*4])
        self.Critic.optimizer.zero_grad()
        output = self.Critic.forward(x)
        critic_loss = F.mse_loss(y, labels)

        critic_loss.backward()
        self.Critic.optimizer.step()

        return critic_loss.item()

    def learn(self, episode):
        batch = 5
        #torch.autograd.set_detect_anomaly(True)
        self.Critic.optimizer.zero_grad()
        self.Actor.optimizer.zero_grad()
        #print(episode)
        #s is the state, in the most simple case it is the f_map
        
        s = np.array(episode["s"]).flatten() #NEED TO INCLUDE THE TRANSMIT POWER FOR THE CRITIC TO GUESS THE SINR HOW COULD YOU NOT SEE THAT LIKE 1 MONTH AGO YOU STUPID CUNT
        #print(f"state in learn {episode['s']}")
        s = torch.tensor(s, requires_grad=True, dtype=float).float().to(self.device) #current state
        r = torch.tensor(episode["r"], requires_grad=True).float().to(self.device) #the embeded objective function (sum-rate, capcity, SINR...)
        d = torch.tensor(episode["d"]).bool().to(self.device) #done, not really necessary
        #s_ = np.array(episode["s_"]).flatten()
        #s_ = torch.tensor(s_, requires_grad=True).float().to(self.device) #new state
        lg_p = torch.tensor(self.log_probs, requires_grad=True, dtype=float).float().to(self.device) #log_probs as given by choose_action
        
        #get critic values for current and next state
        val = self.Critic.forward(s.view([1, self.nb_devices*5]))

        
        #compute the delta
        
        delta = r  - val #+ self.gamma * val_ - val
        #print(f"delta {delta} \nval {val}")
        #print(f"r {r} f_map {s} lg_p {lg_p}")
        critic_loss = delta**2
        critic_loss.backward(retain_graph=True)
        self.Critic.optimizer.step()
        actor_loss = -torch.mean(lg_p.flatten()*delta)
        critic_loss = delta**2
        print(f"--actual reward {np.round(r.detach().numpy(), 2)} \n--predicted value for state {np.round(val.detach().numpy(), 2)}\n--difference of {np.round(delta.detach().numpy(), 4)}")
        
        
        actor_loss.backward(retain_graph=False)
        self.Actor.optimizer.step()
        self.Critic.optimizer.step()

        return actor_loss.item(), critic_loss.item(), val.item(), r.item()
    



    