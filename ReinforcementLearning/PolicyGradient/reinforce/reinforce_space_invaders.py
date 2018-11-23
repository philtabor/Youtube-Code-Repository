import numpy as np
import gym
from cnn_model import PolicyGradientAgent
import matplotlib.pyplot as plt
from utils import plotLearning

def process_obs(observation):
    return np.mean(observation[15:200, 30:125], axis=2)

if __name__ == '__main__':
    agent = PolicyGradientAgent(ALPHA=0.00001, GAMMA=0.95, n_actions=6)
    env = gym.make('SpaceInvaders-v0')    
    score_history = []
    score = 0
    num_episodes = 1000
    last_action = 0
    for i in range(num_episodes):
        print('episode: ', i,'score: ', score)
        done = False
        score = 0        
        observation = env.reset()
        frames = [process_obs(observation)]
        while not done:    
            if len(frames) == 3:
                action = agent.choose_action(frames)
                frames = []
            else:
                action = last_action                                                        
            observation_, reward, done, info = env.step(action)                        
            agent.store_transition(process_obs(observation), action, reward)            
            observation = observation_
            last_action = action
            frames.append(process_obs(observation))
            score += reward
        score_history.append(score)        
        agent.learn()                
    filename = 'alpha00001-rewards-space-invaders.png'
    plotLearning(score_history, filename=filename, window=20)