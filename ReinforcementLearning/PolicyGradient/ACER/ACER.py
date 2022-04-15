from enum import unique
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from collections import deque
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self,alpha,input_dims,fc1_dims,fc2_dims,action_dims):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dims = action_dims
        self.f1 = nn.Linear(self.input_dims,self.fc1_dims)
        self.f2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.f3 = nn.Linear(self.fc2_dims,self.action_dims)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.relu = torch.nn.ReLU()
        
        
        self.to(self.device)

    def forward(self,observation):
        state = torch.tensor(observation).to(self.device)
        x = self.f1(state)
        x = self.relu(x)
        x = self.f2(x)
        x = self.relu(x)
        x = self.f3(x)

        return x
    
class CriticNetwork(nn.Module):
    def __init__(self,alpha,input_dims,f1_dims,f2_dims) -> None:
        super().__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.f1_dims = f1_dims
        self.f2_dims = f2_dims
        self.f1  = nn.Linear(input_dims,f1_dims)
        self.f2 = nn.Linear(f1_dims,f2_dims)
        self.f3 = nn.Linear(f2_dims,1)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.relu = nn.ReLU()

        self.to(self.device)

    def forward(self, observation):
        state = torch.tensor(observation).to(self.device)
        x = self.f1(state)
        x = self.relu(x)
        x = self.f2(x)
        x = self.relu(x)
        x = self.f3(x)   

        return x

class Agent():
    def __init__(self,lr,input_dims,action_dims,gamma=0.99,l1_size=256,l2_size=256,bath_size=32,mem_size=1000000) -> None:
        self.gamma = gamma
        self.batch_size = bath_size
        self.memory = deque(mem_size)
        self.actor = ActorNetwork(lr,input_dims,l1_size,l2_size,action_dims)
        self.critic = CriticNetwork(lr,input_dims,l1_size,l2_size)

    def push_transaction(self,state,reward,next_state,done):
        self.memory.append((state,reward,next_state,done))
    
    def choose_action(self,observation):
        state = torch.tensor(observation).to(self.actor.device)
        probs = self.actor.forward(state)
        probs = F.softmax(probs)
        action_dist = Categorical(probs)
        action = action_dist.sample()

        return action.item()

    def learn(self):
        if len(self.memory)<self.batch_size: return
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()

        transitions = random.sample(self.memory,self.batch_size)
        states = torch.tensor([t[0] for t in transitions]).to(self.critic.device)
        # probs = torch.tensor([t[1] for t in transitions]).to(self.critic.device)
        rewards = torch.tensor([t[2] for t in transitions]).to(self.critic.device)
        next_states = torch.tensor([t[3] for t in transitions]).to(self.critic.device)
        dones = torch.tensor([t[4] for t in transitions]).to(self.critic.device)

        Vs = self.critic.forward(states)
        next_Vs = self.critic.forward(next_states)

        next_Vs[dones] = 0

        delta = rewards+self.gamma*next_Vs

        actor_loss = -torch.mean(probs*(delta -  Vs))
        critic_loss = F.mse_loss(delta,Vs)
        
        (actor_loss+critic_loss).backward()
        
        self.actor.optimizer.step()
        self.critic.optimizer.step()

        







        


