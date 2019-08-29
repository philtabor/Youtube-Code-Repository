import os
# for keras the CUDA commands must come before importing the keras libraries
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import gym
from gym import wrappers
import numpy as np
from ddqn_keras import DDQNAgent
from utils import plotLearning

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
                  batch_size=64, input_dims=8)
    n_games = 500
    #ddqn_agent.load_model()
    ddqn_scores = []
    eps_history = []
    #env = wrappers.Monitor(env, "tmp/lunar-lander-ddqn-2",
    #                         video_callable=lambda episode_id: True, force=True)

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()
        eps_history.append(ddqn_agent.epsilon)

        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            ddqn_agent.save_model()

    filename = 'lunarlander-ddqn.png'

    x = [i+1 for i in range(n_games)]
    plotLearning(x, ddqn_scores, eps_history, filename)
