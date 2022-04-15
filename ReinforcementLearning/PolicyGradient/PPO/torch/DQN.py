import torch
from torch import nn
import gym
from collections import deque
import itertools
import numpy as np
import random

GAMMA = 0.99
BATH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000

# the neural network input is state output is q values for each actions
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_feature = int(np.prod(env.observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_feature, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
            # nn.Linear(in_feature,env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self(state_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action

env = gym.make('CartPole-v0')
replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0
mean_reward_history = []

online_net = Network(env)
target_net = Network(env)
target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

state = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    transition = (state, action, reward, done, next_state)
    replay_buffer.append(transition)
    state = next_state

    if done:
        state = env.reset()

# training loop
state = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(state)

    next_state, reward, done, _ = env.step(action)
    transition = (state, action, reward, done, next_state)
    replay_buffer.append(transition)
    state = next_state

    episode_reward += reward
    if done:
        state = env.reset()
        reward_buffer.append(episode_reward)
        episode_reward = 0.0

# play the video
#     if len(reward_buffer) >= 100:
#         if np.mean(reward_buffer) >= 190:
#             while True:
#                 action = online_net.act(state)
#                 state, _, done, _ = env.step(action)
#                 env.render()
#                 if done:
#                     env.reset()

    # start gradient step
    transitions = random.sample(replay_buffer, BATH_SIZE)

    states = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    next_states = np.asarray([t[4] for t in transitions])

    states_t = torch.as_tensor(states, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32)

    # Compute target
    target_q_values = target_net(next_states_t).detach()
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
    targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values

    # computing loss
    q_values = online_net(states_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    huber_loss = nn.SmoothL1Loss()
    loss = huber_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
        print()
        print('step', step)
        print('Avg Rew', np.mean(reward_buffer))
    mean_reward_history.append(np.mean(reward_buffer))
    if step>=300:break
import pandas as pd
pd.DataFrame(mean_reward_history).to_csv('result_DQN.csv')
