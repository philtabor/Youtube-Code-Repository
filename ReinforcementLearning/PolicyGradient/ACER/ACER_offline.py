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

        probs = F.softmax(x)
        action_dist = Categorical(probs)
        return action_dist
    
class CriticNetwork(nn.Module):
    def __init__(self,alpha,input_dims,f1_dims,f2_dims,chkpt_dir='model') -> None:
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_acer')
        self.alpha = alpha
        self.input_dims = input_dims
        self.f1_dims = f1_dims
        self.f2_dims = f2_dims
        self.s_layer  = nn.Linear(input_dims,f1_dims)
        self.a_layer = nn.Linear(1,f1_dims)
        self.f2 = nn.Linear(f1_dims,f2_dims)
        self.f3 = nn.Linear(f2_dims,1)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.relu = nn.ReLU()
        

        self.to(self.device)

    def forward(self, observation,action):
        # state = torch.tensor(observation,dtype=torch.float).to(self.device)
        # action = torch.tensor(action,dtype=torch.float).to(self.device).unsqueeze(1)
        s = self.s_layer(observation.float())
        s = self.relu(s)
        
        a = self.a_layer(action.float())
        a = self.relu(a)

        x = torch.add(s,a)
        x = self.f2(x)
        x = self.relu(x)
        x = self.f3(x)   

        return x

class Agent():
    def __init__(self,lr,tau,input_dims,action_dims,gamma=0.99,l1_size=256,l2_size=256,bath_size=32,mem_size=50000) -> None:
        self.gamma = gamma
        self.tau = tau
        self.batch_size = bath_size
        self.ephoch = 10
        self.memory = deque(maxlen=mem_size)
        self.actor = ActorNetwork(lr,input_dims,l1_size,l2_size,action_dims)
        self.actor_target = ActorNetwork(lr,input_dims,l1_size,l2_size,action_dims)
        self.critic = CriticNetwork(lr,input_dims,l1_size,l2_size)
        self.critic_target = CriticNetwork(lr,input_dims,l1_size,l2_size)

    def push_transaction(self,state,reward,next_state,done):
        self.memory.append((state,reward,next_state,done))
    
    def choose_action(self,observation):
        state = torch.tensor(observation,dtype=torch.float).to(self.actor.device)
        action_dist = self.actor_target.forward(state)
        action = action_dist.sample()
        return action

    def load_model(self,):
        print('....loading model....')
        self.actor.load_state_dict(torch.load(self.actor.checkpoint_file))
        self.critic.load_state_dict(torch.load(self.critic.checkpoint_file))

    def save_model(self,):
        print('....saving mdoel....')
        torch.save(self.actor.state_dict(),self.actor.checkpoint_file)
        torch.save(self.critic.state_dict(),self.critic.checkpoint_file)

    def update(self):
        params_actor_target = dict(self.actor_target.named_parameters())
        params_critic_target = dict(self.critic_target.named_parameters())
        params_actor = dict(self.actor.named_parameters())
        params_critic = dict(self.critic.named_parameters())

        for name in params_actor:
            params_actor_target[name] = self.tau * params_actor_target[name] + (1-self.tau)*params_actor[name]
        for name in params_critic:
            params_critic_target[name] = self.tau * params_critic_target[name] + (1-self.tau)*params_critic[name]
        
        self.actor_target.load_state_dict(params_actor_target)
        self.critic_target.load_state_dict(params_critic_target)

    
    def learn(self):
        # if len(self.memory)<self.batch_size: return
        transitions = self.memory
        states = torch.tensor([t[0] for t in transitions]).to(self.critic.device)
        actions = torch.tensor([t[1].tolist() for t in transitions]).to(self.critic.device).unsqueeze(1)
        rewards = torch.tensor([t[2] for t in transitions]).to(self.critic.device).unsqueeze(1)
        next_states = torch.tensor([t[3] for t in transitions]).to(self.critic.device)
        dones = torch.tensor([t[4] for t in transitions]).to(self.critic.device)
        
        for _ in range(self.ephoch):
            self.critic.optimizer.zero_grad()
            self.actor.optimizer.zero_grad()
            dists = self.actor.forward(states)
            next_actions = dists.sample()-0.5
            # log_pobs = dist.log_prob(actions)

            Qs = self.critic.forward(states,actions)
            next_Qs = self.critic_target.forward(next_states,next_actions.unsqueeze(1)).detach()
            next_Qs[dones] = 0
            delta = rewards+self.gamma*next_Qs

            

            critic_loss = F.mse_loss(delta,Qs)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.critic.eval()
            Qs = self.critic.forward(states,self.actor.forward(states).sample().unsqueeze(1))
            actor_loss = -torch.mean(Qs)
            actor_loss.backward()
            self.actor.optimizer.step()
            self.critic.train()
            self.update()


