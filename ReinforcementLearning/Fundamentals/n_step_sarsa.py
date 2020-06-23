import numpy as np
import gym

poleThetaSpace = np.linspace(-0.209, 0.209, 10)
poleThetaVelSpace = np.linspace(-4, 4, 10)
cartPosSpace = np.linspace(-2.4, 2.4, 10)
cartVelSpace = np.linspace(-4, 4, 10)

def get_state(observation):
    cartX, cartXdot, cartTheta, cartThetaDot = observation
    cartX = int(np.digitize(cartX, cartPosSpace))
    cartXdot = int(np.digitize(cartXdot, cartVelSpace))
    cartTheta = int(np.digitize(cartTheta, poleThetaSpace))
    cartThetaDot = int(np.digitize(cartThetaDot, poleThetaVelSpace))

    return (cartX, cartXdot, cartTheta, cartThetaDot)

def choose_action(q, obs, eps, n_actions=2):
    state = get_state(obs)
    if np.random.random() < eps:
        action = np.random.choice([i for i in range(n_actions)])
    else:
        action_values = [q[(state, a)] for a in range(n_actions)]
        action = np.argmax(action_values)
    return action

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0

    states = []
    for i in range(len(cartPosSpace)+1):
        for j in range(len(cartVelSpace)+1):
            for k in range(len(poleThetaSpace)+1):
                for l in range(len(poleThetaVelSpace)+1):
                    states.append((i,j,k,l))

    Q = {}
    for s in states:
        for a in range(2):
            Q[(s, a)] = 0.0

    n = 16
    state_memory = np.zeros((n, 4))
    action_memory = np.zeros(n)
    reward_memory = np.zeros(n)

    scores = []
    n_episodes = 50000
    for i in range(n_episodes):
        done = False
        score = 0
        t = 0
        T = np.inf
        observation = env.reset()
        action = choose_action(Q, observation, epsilon)
        action_memory[t%n] = action
        state_memory[t%n] = observation
        while not done:
            observation, reward, done, info = env.step(action)
            score += reward
            state_memory[(t+1)%n] = observation
            reward_memory[(t+1)%n] = reward
            if done:
                T = t + 1
                #print('episode ends at step', t)
            action = choose_action(Q, observation, epsilon)
            action_memory[(t+1)%n] = action
            tau = t - n + 1
            if tau >= 0:
                G = [gamma**(j-tau-1)*reward_memory[j%n] \
                        for j in range(tau+1, min(tau+n, T)+1)]
                G = np.sum(G)
                if tau + n < T:
                    s = get_state(state_memory[(tau+n)%n])
                    a = int(action_memory[(tau+n)%n])
                    G += gamma**n * Q[(s,a)]
                s = get_state(state_memory[tau%n])
                a = action_memory[tau%n]
                Q[(s,a)] += alpha*(G-Q[(s,a)])
            #print('tau ', tau, '| Q %.2f' % \
            #        Q[(get_state(state_memory[tau%n]), action_memory[tau%n])])

            t += 1

        for tau in range(t-n+1, T):
            G = [gamma**(j-tau-1)*reward_memory[j%n] \
                    for j in range(tau+1, min(tau+n, T)+1)]
            G = np.sum(G)
            if tau + n < T:
                s = get_state(state_memory[(tau+n)%n])
                a = int(action_memory[(tau+n)%n])
                G += gamma**n * Q[(s,a)]
            s = get_state(state_memory[tau%n])
            a = action_memory[tau%n]
            Q[(s,a)] += alpha*(G-Q[(s,a)])
            #print('tau ', tau, '| Q %.2f' % \
            #    Q[(get_state(state_memory[tau%n]), action_memory[tau%n])])
        scores.append(score)
        avg_score = np.mean(scores[-1000:])
        epsilon = epsilon -2 / n_episodes if epsilon > 0 else 0
        if i % 1000 == 0:
            print('episode ', i, 'avg_score %.1f' % avg_score,
                    'epsilon %.2f' % epsilon)

