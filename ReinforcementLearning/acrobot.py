import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import pickle

theta_space = np.linspace(-1, 1, 10)
theta_dot_space = np.linspace(-10, 10, 10)

def get_state(observation):
    cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot = \
        observation
    c_th1 = int(np.digitize(cos_theta1, theta_space))
    s_th1 = int(np.digitize(sin_theta1, theta_space))
    c_th2 = int(np.digitize(cos_theta2, theta_space))
    s_th2 = int(np.digitize(sin_theta2, theta_space))
    th1_dot = int(np.digitize(theta1_dot, theta_dot_space))
    th2_dot = int(np.digitize(theta2_dot, theta_dot_space))

    return (c_th1, s_th2, c_th2, s_th2, th1_dot, th2_dot)

def maxAction(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action

if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    n_games = 100
    alpha = 0.1
    gamma = 0.99
    eps = 0

    action_space = [0, 1, 2]

    states = []
    for c1 in range(11):
        for s1 in range(11):
            for c2 in range(11):
                for s2 in range(11):
                    for dot1 in range(11):
                        for dot2 in range(11):
                            states.append((c1, s1, c2, s2, dot1, dot2))
    """
    Q = {}
    for state in states:
        for action in action_space:
            Q[state, action] = 0
    """
    pickle_in = open('acrobot.pkl', 'rb')
    Q = pickle.load(pickle_in)
    env = wrappers.Monitor(env, "tmp/acrobot", video_callable=lambda episode_id: True, force=True)
    eps_rewards = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        if i % 1 == 0:
            print('episode ', i, 'score ', eps_rewards, 'eps', eps)
        observation = env.reset()
        state = get_state(observation)
        done = False
        action = env.action_space.sample() if np.random.random() < eps else \
                 maxAction(Q, state)
        eps_rewards = 0
        while not done:
            """
            print(observation)
            action = env.action_space.sample()
            """
            observation_, reward, done, info = env.step(action)
            state_ = get_state(observation_)
            action_ = maxAction(Q, state_)
            eps_rewards += reward
            Q[state, action] = Q[state,action] + \
                    alpha*(reward + gamma*Q[state_,action_] - Q[state,action])
            state = state_
            action = action_
        total_rewards[i] = eps_rewards
        eps = eps - 2 / n_games if eps > 0.01 else 0.01

    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.show()

    f = open("acrobot.pkl","wb")
    pickle.dump(Q,f)
    f.close()
