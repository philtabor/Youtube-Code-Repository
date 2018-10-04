import gym
from deepQModel import deepQNetwork
from utils import plotLearning
import matplotlib.pyplot as plt
import torch as T
import numpy as np 

if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    brain = deepQNetwork(gamma=0.9, epsilon=1.0, 
                         alpha=0.003, maxMemorySize=2500)   
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            if done and info['ale.lives'] == 0:
                reward = -100                  
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward, 
                                np.mean(observation_[15:200,30:125], axis=2))
            observation = observation_
    print('done initializing memory')

    scores = []
    epsHistory = []
    numGames = 250
    for i in range(numGames):
        print('starting game ', i+1, 'epsilon: ', brain.EPSILON)
        epsHistory.append(brain.EPSILON)        
        done = False
        observation = env.reset()
        frames = [np.sum(observation[15:200,30:125], axis=2)]
        score = 0        
        while not done:
            if len(frames) == 3:
                action = brain.chooseAction(frames)
                frames = []
            else:
                action = 0
            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(observation_[15:200,30:125], axis=2))
            if done and info['ale.lives'] == 0:
                reward = -100 
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward, 
                                  np.mean(observation_[15:200,30:125], axis=2))
            observation = observation_            
            brain.learn(batch_size=24)
            #env.render()
        scores.append(score)
        print('score:',score)
    x = [i+1 for i in range(numGames)]
    fileName = str(numGames) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
               'Alpha' + str(brain.ALPHA) + 'Memory' + str(brain.memSize)+ '.png'    
    plotLearning(x, scores, epsHistory, fileName)