import argparse
import gym
from dqn_torch import Agent
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-bs', type=int, default=1000)
    parser.add_argument('-cer', type=bool, default=False)
    # if you supply it, then true
    args = parser.parse_args()

    env = gym.make('LunarLander-v2')
    combined = args.cer
    buffer_size = args.bs

    agent = Agent(gamma=0.99, epsilon=0.1, batch_size=64, n_actions=4,
                  eps_end=0.1, input_dims=[8], lr=0.001,
                  max_mem_size=buffer_size, combined=combined)

    scores = []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.memory.store_transition(observation, action, reward,
                                          observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)

        avg_score = np.mean(scores[-100:])

        print('combined {} episode {} score {:.0f} avg score {:.0f} eps {:.2f}'
              .format(combined, i, score, avg_score, agent.epsilon))

    if combined:
        fname = 'CER_const_eps_' + str(buffer_size) + '.npy'
    else:
        fname = 'VER_const_eps_' + str(buffer_size) + '.npy'
    np.save(fname, np.array(scores))
