import numpy as np


class ReplayMemory:
    def __init__(self, input_dims, max_mem, batch_size, combined=False):
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.combined = combined
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def sample_memory(self):
        offset = 1 if self.combined else 0
        max_mem = min(self.mem_cntr, self.mem_size) - offset
        batch = np.random.choice(max_mem, self.batch_size-offset,
                                 replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        if self.combined:
            index = self.mem_cntr % self.mem_size - 1
            last_action = self.action_memory[index]
            last_state = self.state_memory[index]
            last_new_state = self.new_state_memory[index]
            last_reward = self.reward_memory[index]
            last_terminal = self.terminal_memory[index]

            actions = np.append(self.action_memory[batch], last_action)
            states = np.vstack((self.state_memory[batch], last_state))
            new_states = np.vstack((self.new_state_memory[batch],
                                   last_new_state))
            rewards = np.append(self.reward_memory[batch], last_reward)
            terminals = np.append(self.terminal_memory[batch], last_terminal)

        return states, actions, rewards, new_states, terminals

    def is_sufficient(self):
        return self.mem_cntr > self.batch_size
