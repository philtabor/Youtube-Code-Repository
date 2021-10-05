import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, tau=0.98):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.tau = tau

        self.input = nn.Linear(*input_dims, 256)
        self.dense = nn.Linear(256, 256)

        self.gru = nn.GRUCell(256, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def forward(self, state, hx):
        x = F.relu(self.input(state))
        x = F.relu(self.dense(x))
        hx = self.gru(x, (hx))

        pi = self.pi(hx)
        v = self.v(hx)

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy()[0], v, log_prob, hx

    def calc_R(self, done, rewards, values):
        values = T.cat(values).squeeze()
        if len(values.size()) == 1:  # batch of states
            R = values[-1] * (1-int(done))
        elif len(values.size()) == 0:  # single state
            R = values*(1-int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, 
                                dtype=T.float).reshape(values.size())
        return batch_return

    def calc_loss(self, new_states, hx, done,
                  rewards, values, log_probs, r_i_t=None):
        if r_i_t is not None:
            rewards += r_i_t.detach().numpy()
        returns = self.calc_R(done, rewards, values)
        next_v = T.zeros(1, 1) if done else self.forward(T.tensor([new_states],
                                         dtype=T.float), hx)[1]

        values.append(next_v.detach())
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)

        delta_t = rewards + self.gamma*values[1:] - values[:-1]
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps-t):
                temp = (self.gamma*self.tau)**k*delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)

        actor_loss = -(log_probs*gae).sum()
        entropy_loss = (-log_probs*T.exp(log_probs)).sum()
        # [T] vs ()
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)

        total_loss = actor_loss + critic_loss - 0.01*entropy_loss
        return total_loss
