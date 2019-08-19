import gym
import numpy as np
import matplotlib.pyplot as plt

ACTION_DICT = {0: 'NOOP', 1: 'FIRE', 2:'RIGHT', 3:'LEFT'}

def preprocess(observation):
    observation = observation / 255
    return np.mean(observation[30:,:], axis=2).reshape(180,160)

def stack_frames(stacked_frames, frame, stack_size, actions, action):
    if stacked_frames is None:
        stacked_frames = np.zeros((*frame.shape, stack_size))
        actions = np.zeros(stack_size)
        for idx in range(stack_size):
            stacked_frames[:,:,idx] = frame
    else:
        stacked_frames[:,:,0:stack_size-1] = stacked_frames[:,:,1:]
        stacked_frames[:,:,stack_size-1] = frame
        actions[0:stack_size-1] = actions[1:]
        actions[stack_size-1] = action
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    ax1.imshow(stacked_frames[:,:,0])
    ax1.set_title(ACTION_DICT[actions[0]])
    ax2.imshow(stacked_frames[:,:,1])
    ax2.set_title(ACTION_DICT[actions[1]])
    ax3.imshow(stacked_frames[:,:,2])
    ax3.set_title(ACTION_DICT[actions[2]])
    ax4.imshow(stacked_frames[:,:,3])
    ax4.set_title(ACTION_DICT[actions[3]])
    plt.show()

    return actions, stacked_frames

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    stack_size = 4

    for i in range(10):
        done = False
        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        actions=None
        actions, stacked_frames = stack_frames(stacked_frames, observation,
                                               stack_size, actions, 0)
        while not done:
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            actions, stacked_frames_ = stack_frames(stacked_frames,
                                           preprocess(observation_), stack_size,
                                           actions, action)
