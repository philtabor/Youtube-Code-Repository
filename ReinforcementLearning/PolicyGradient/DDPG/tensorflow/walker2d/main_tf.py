from ddpg_orig_tf import Agent
import gym
import numpy as np
from utils import plotLearning
from gym import wrappers
import os

#tf.set_random_seed(0)
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    env = gym.make('BipedalWalker-v2')
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[24], tau=0.001, env=env,
                  batch_size=64,  layer1_size=400, layer2_size=300, n_actions=4,
                  chkpt_dir='tmp/ddpg')
    np.random.seed(0)
    #agent.load_models()
    #env = wrappers.Monitor(env, "tmp/walker2d",
    #                            video_callable=lambda episode_id: True, force=True)
    score_history = []
    for i in range(5000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            env.render()
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        if i % 25 == 0:
            agent.save_models()
    filename = 'WalkerTF-alpha00005-beta0005-400-300-original-5000games-testing.png'
    plotLearning(score_history, filename, window=100)
