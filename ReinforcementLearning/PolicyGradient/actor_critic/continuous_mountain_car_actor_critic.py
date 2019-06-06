import numpy as np
import gym
from actor_critic_continuous import Agent
import matplotlib.pyplot as plt
from utils import plotLearning
from gym import wrappers


if __name__ == '__main__':
    agent = Agent(alpha=0.000005, beta=0.00001, input_dims=[2], gamma=0.99,
                  layer1_size=256, layer2_size=256)

    env = gym.make('MountainCarContinuous-v0')
    score_history = []
    num_episodes = 100
    for i in range(num_episodes):
        #env = wrappers.Monitor(env, "tmp/mountaincar-continuous-trained-1",
        #                        video_callable=lambda episode_id: True, force=True)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = np.array(agent.choose_action(observation)).reshape((1,))
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode: ', i,'score: %.2f' % score)
    filename = 'mountaincar-continuous-old-actor-critic-alpha000005-256x256fc-100games.png'
    plotLearning(score_history, filename=filename, window=20)
