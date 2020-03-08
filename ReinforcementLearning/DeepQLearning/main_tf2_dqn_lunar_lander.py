from simple_dqn_tf2 import Agent
import numpy as np
import gym
from utils import plotLearning
import tensorflow as tf

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    lr = 0.001
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, 
                input_dims=env.observation_space.shape,
                n_actions=env.action_space.n, mem_size=1000000, batch_size=64,
                epsilon_end=0.01)
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' % score,
                'average_score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    filename = 'lunarlander_tf2.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)
