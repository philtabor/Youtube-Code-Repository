import numpy as np
import gym
from actor_critic_discrete import Agent
import matplotlib.pyplot as plt
from utils import plotLearning
from gym import wrappers


if __name__ == '__main__':
    agent = Agent(alpha=0.0001, beta=0.0005, input_dims=[4], gamma=0.99,
                  n_actions=2, layer1_size=32, layer2_size=32)

    env = gym.make('CartPole-v1')
    score_history = []
    score = 0
    num_episodes = 2500
    for i in range(num_episodes):
        print('episode: ', i,'score: %.3f' % score)


        #env = wrappers.Monitor(env, "tmp/cartpole-untrained",
        #                            video_callable=lambda episode_id: True, force=True)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        score_history.append(score)

    filename = 'cartpole-discrete-actor-critic-alpha0001-beta0005-32x32fc-1500games.png'
    plotLearning(score_history, filename=filename, window=10)
