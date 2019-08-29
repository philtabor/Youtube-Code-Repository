import numpy as np
import gym
from reinforce_cnn_tf import PolicyGradientAgent
from utils import plotLearning
from gym import wrappers


def preprocess(observation):
    return np.mean(observation[15:200, 30:125], axis=2)


def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx,:] = frame
    else:
        stacked_frames[0:buffer_size-1,:] = stacked_frames[1:,:]
        stacked_frames[buffer_size-1, :] = frame

    return stacked_frames

if __name__ == '__main__':
    load_checkpoint = False
    agent = PolicyGradientAgent(ALPHA=0.001, GAMMA=0.9, n_actions=6, fc1=256,
                                chkpt_dir='tmp/checkpoint-newG-0p001', gpu={'GPU':1})
    filename = 'space-invaders-alpha001-newGcalc.png'
    print('will use ', filename, ' and ', agent.gpu)
    if load_checkpoint:
        agent.load_checkpoint()
    env = gym.make('SpaceInvaders-v0')
    score_history = []
    score = 0
    num_episodes = 1000
    stack_size = 4
    #env = wrappers.Monitor(env, "tmp/space-invaders-newG-0p003",
    #                        video_callable=lambda episode_id: True, force=True)
    for i in range(num_episodes):
        done = False

        avg_score = np.mean(score_history[max(0, i-20):(i+1)])
        if i % 20 == 0 and i > 0:
            print('episode: ', i,'score: ', score, ' average score %.3f' % avg_score)
            plotLearning(score_history, filename=filename, window=20)
        else:
            print('episode: ', i,'score: ', score)
        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        stacked_frames = stack_frames(stacked_frames, observation, stack_size)
        score = 0
        while not done:
            action = agent.choose_action(stacked_frames)
            observation, reward, done, info = env.step(action)
            observation = preprocess(observation)
            stacked_frames = stack_frames(stacked_frames, observation, stack_size)
            agent.store_transition(observation, action, reward)

            score += reward
        score_history.append(score)

        if i % 10 == 0:
            agent.learn()
            agent.save_checkpoint()
    plotLearning(score_history, filename=filename, window=20)
