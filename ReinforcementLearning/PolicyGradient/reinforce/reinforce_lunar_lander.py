import numpy as np
import gym
from model import PolicyGradientAgent
import matplotlib.pyplot as plt
from utils import plotLearning

if __name__ == '__main__':
    agent = PolicyGradientAgent(ALPHA=0.001, input_dims=8, n_actions=4)
    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episodes = 3000

    for i in range(num_episodes):
        print('episode: ', i,'score: ', score)
        done = False
        score = 0        
        observation = env.reset()
        while not done:    
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)                        
            agent.store_transition(observation, action, reward)            
            observation = observation_               
            score += reward
        score_history.append(score)        
        agent.learn()                
    filename = 'alpha001-rewards-lunar-lander.png'
    plotLearning(score_history, filename=filename, window=50)