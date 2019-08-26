import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
from reinforce_keras import Agent
from utils import plotLearning

if __name__ == '__main__':
    agent = Agent(ALPHA=0.0005, input_dims=8, GAMMA=0.99,
                  n_actions=4, layer1_size=64, layer2_size=64)

    env = gym.make('LunarLander-v2')
    score_history = []

    num_episodes = 2000

    for i in range(num_episodes):
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

        _ = agent.learn()
        print('episode: ', i,'score: %.1f' % score,
            'average score %.1f' % np.mean(score_history[max(0, i-100):(i+1)]))

    filename = 'lunar-lander-keras-64x64-alpha0005-2000games.png'
    plotLearning(score_history, filename=filename, window=100)
