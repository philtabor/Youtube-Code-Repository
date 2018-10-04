import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class deepQNetwork(nn.Module):
    def __init__(self, gamma, epsilon, alpha, 
                 maxMemorySize, actionSpace=[0,1,2,3,4,5]):
        super(deepQNetwork, self).__init__()
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.ALPHA = alpha
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.memory = []
        self.memCntr = 0
        #self.conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=1)
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        #self.fc1 = nn.Linear(128*23*16, 512)
        self.fc1 = nn.Linear(128*19*8, 512)
        self.fc2 = nn.Linear(512, 6)
        #self.optimizer = optim.SGD(self.parameters(), lr=self.ALPHA, momentum=0.9)
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')     
        self.to(self.device)        

    def forward(self, observation):
        observation = T.Tensor(observation).to(self.device)        
        #observation = observation.view(-1, 3, 210, 160).to(self.device)        
        observation = observation.view(-1, 1, 185, 95).to(self.device)        
        observation = F.relu(self.conv1(observation)).to(self.device)        
        observation = F.relu(self.conv2(observation)).to(self.device)        
        observation = F.relu(self.conv3(observation)).to(self.device)    
        #observation = observation.view(-1, 128*23*16).to(self.device)        
        observation = observation.view(-1, 128*19*8).to(self.device)        
        observation = F.relu(self.fc1(observation)).to(self.device)        
        actions = self.fc2(observation).to(self.device)        
        return actions

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:            
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr += 1
        
    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.forward(observation)
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1]).item()            
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action
    
    def learn(self, batch_size):
        self.optimizer.zero_grad()
        if self.memCntr+batch_size < self.memSize:            
            memStart = int(np.random.choice(range(self.memCntr-batch_size-1)))
        else:
            memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
        miniBatch=self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        Qpred = self.forward(list(memory[:,0][:])).to(self.device)
        Qnext = self.forward(list(memory[:,3][:])).to(self.device)

        maxA = T.argmax(Qnext, dim=1).to(self.device) 
        Qtarget = Qpred
        rewards = T.Tensor(list(memory[:,2])).to(self.device)        
        Qtarget[:,maxA] = rewards + self.GAMMA*T.max(Qnext[1])
        
        if self.steps > 25000:
            if self.EPSILON - 5e-6 > 0.05:
                self.EPSILON -= 5e-6
            else:
                self.EPSILON = 0

        Qpred.requires_grad_()
        loss = self.loss(Qtarget, Qpred).to(self.device)
        loss.backward()
        self.optimizer.step()       