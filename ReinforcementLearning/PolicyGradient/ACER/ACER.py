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
    def __init__(self,alpha,input_dims,fc1_dims,fc2_dims,action_dims,chkpt_dir='model'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_acer')
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
        state = torch.tensor(observation,dtype=torch.float).to(self.device)
        x = self.f1(state)
        x = self.relu(x)
        x = self.f2(x)
        x = self.relu(x)
        x = self.f3(x)

        return x
    
class CriticNetwork(nn.Module):
    def __init__(self,alpha,input_dims,f1_dims,f2_dims,chkpt_dir='model') -> None:
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_acer')
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
        state = torch.tensor(observation,dtype=torch.float).to(self.device)
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
        self.memory = deque(maxlen=mem_size)
        self.actor = ActorNetwork(lr,input_dims,l1_size,l2_size,action_dims)
        self.actor_target = ActorNetwork(lr,input_dims,l1_size,l2_size,action_dims)
        self.critic = CriticNetwork(lr,input_dims,l1_size,l2_size)
        self.critic_target = CriticNetwork(lr,input_dims,l1_size,l2_size)

    def push_transaction(self,state,reward,next_state,done):
        self.memory.append((state,reward,next_state,done))
    
    def choose_action(self,observation):
        state = torch.tensor(observation,dtype=torch.float).to(self.actor.device)
        probs = self.actor_target.forward(state)
        probs = F.softmax(probs)
        action_dist = Categorical(probs)
        action = action_dist.sample()

        return action.item(),torch.log(probs).detach().cpu().numpy()

    def load_model(self,):
        print('....loading model....')
        self.actor.load_state_dict(torch.load(self.actor.checkpoint_file))
        self.critic.load_state_dict(torch.load(self.critic.checkpoint_file))

    def save_model(self,):
        print('....saving mdoel....')
        torch.save(self.actor.state_dict(),self.actor.checkpoint_file)
        torch.save(self.critic.state_dict(),self.critic.checkpoint_file)

    def learn(self):
        if len(self.memory)<self.batch_size: return
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()

        transitions = random.sample(self.memory,self.batch_size)
        states = torch.tensor([t[0] for t in transitions]).to(self.critic.device)
        probs = torch.tensor([t[1] for t in transitions]).to(self.critic.device)
        rewards = torch.tensor([t[1] for t in transitions]).to(self.critic.device)
        next_states = torch.tensor([t[2] for t in transitions]).to(self.critic.device)
        dones = torch.tensor([t[3] for t in transitions]).to(self.critic.device)

        Vs = self.critic.forward(states)
        next_Vs = self.critic_target.forward(next_states)

        next_Vs[dones] = 0

        delta = rewards+self.gamma*next_Vs

        actor_loss = -torch.mean(probs*(delta -  Vs))
        critic_loss = F.mse_loss(delta,Vs)
        
        (actor_loss+critic_loss).backward()
        
        self.actor.optimizer.step()
        self.critic.optimizer.step()

        







        


