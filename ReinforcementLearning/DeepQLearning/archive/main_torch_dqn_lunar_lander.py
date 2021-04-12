import gym
from simple_dqn_torch import DeepQNetwork, Agent
from utils import plotLearning
import numpy as np
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  input_dims=[8], alpha=0.003)

    scores = []
    eps_history = []
    num_games = 500
    score = 0
    # uncomment the line below to record every episode.
    #env = wrappers.Monitor(env, "tmp/space-invaders-1",
    #video_callable=lambda episode_id: True, force=True)
    for i in range(num_games):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % brain.EPSILON)
        else:
            print('episode: ', i,'score: ', score)
        eps_history.append(brain.EPSILON)
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = brain.chooseAction(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            brain.storeTransition(observation, action, reward, observation_,
                                  done)
            observation = observation_
            brain.learn()

        scores.append(score)

    x = [i+1 for i in range(num_games)]
    filename = str(num_games) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
               'Alpha' + str(brain.ALPHA) + 'Memory' + \
                str(brain.Q_eval.fc1_dims) + '-' + str(brain.Q_eval.fc2_dims) +'.png'
    plotLearning(x, scores, eps_history, filename)
