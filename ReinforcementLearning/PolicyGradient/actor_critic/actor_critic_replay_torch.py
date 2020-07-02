import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.log_probs = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, log_prob, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.log_probs[index] = log_prob
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        probs = self.log_probs[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, probs, rewards, states_, terminal

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)

class Agent():
    def __init__(self, lr, input_dims, n_actions, gamma=0.99,
                 l1_size=256, l2_size=256, batch_size=32,
                 mem_size=1000000):
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.actor_critic = ActorCriticNetwork(lr, input_dims, l1_size,
                                    l2_size, n_actions=n_actions)
        self.log_probs = []

    def store_transition(self, state, prob, reward, state_, done):
        self.memory.store_transition(state, prob, reward, state_, done)

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)

        return action.item(), log_probs

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.actor_critic.optimizer.zero_grad()

        state, prob, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.actor_critic.device)
        probs = T.tensor(prob).to(self.actor_critic.device)
        rewards = T.tensor(reward).to(self.actor_critic.device)
        dones = T.tensor(done).to(self.actor_critic.device)
        states_ = T.tensor(new_state).to(self.actor_critic.device)

        _, critic_value_ = self.actor_critic.forward(states_)
        _, critic_value = self.actor_critic.forward(states)

        critic_value_[dones] = 0.0

        delta = rewards + self.gamma*critic_value_

        actor_loss = -T.mean(probs*(delta-critic_value))
        critic_loss = F.mse_loss(delta, critic_value)

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()
