import gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    EPS = 0.05
    GAMMA = 1.0

    agentSumSpace = [i for i in range(4, 22)]
    dealerShowCardSpace = [i+1 for i in range(10)]
    agentAceSpace = [False, True]
    actionSpace = [0, 1] # stick or hit
    stateSpace = []
    
    Q = {}
    C = {}
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for ace in agentAceSpace:
                for action in actionSpace:
                    Q[((total, card, ace), action)] = 0
                    C[((total, card, ace), action)] = 0
                stateSpace.append((total, card, ace))

    targetPolicy = {}
    for state in stateSpace:
        values = np.array([Q[(state, a)] for a in actionSpace ])
        best = np.random.choice(np.where(values==values.max())[0])        
        targetPolicy[state] = actionSpace[best]

    numEpisodes = 1000000
    for i in range(numEpisodes):        
        memory = []
        if i % 100000 == 0:
            print('starting episode', i)
        behaviorPolicy = {}
        for state in stateSpace:
            rand = np.random.random()
            if rand < 1 - EPS:
                behaviorPolicy[state] = [targetPolicy[state]]
            else:
                behaviorPolicy[state] = actionSpace
        observation = env.reset()
        done = False
        while not done:
            action = np.random.choice(behaviorPolicy[observation])
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1], observation[2], action, reward))
            observation = observation_
        memory.append((observation[0], observation[1], observation[2], action, reward))    

        G = 0
        W = 1
        last = True
        for playerSum, dealerCard, usableAce, action, reward in reversed(memory):
            sa = ((playerSum, dealerCard, usableAce), action)
            if last:
                last = False
            else:
                C[sa] += W
                Q[sa] += (W / C[sa])*(G-Q[sa])                
                values = np.array([Q[(state, a)] for a in actionSpace ])
                best = np.random.choice(np.where(values==values.max())[0])        
                targetPolicy[state] = actionSpace[best]
                if action != targetPolicy[state]:
                    break
                if len(behaviorPolicy[state]) == 1:
                    prob = 1 - EPS
                else:
                    prob = EPS / len(behaviorPolicy[state])             
                W *= 1/prob                             
            G = GAMMA*G + reward
        if EPS - 1e-7 > 0:
            EPS -= 1e-7
        else:
            EPS = 0
    numEpisodes = 1000
    rewards = np.zeros(numEpisodes)
    totalReward = 0
    wins = 0
    losses = 0
    draws = 0
    print('getting ready to test target policy')   
    for i in range(numEpisodes):
        observation = env.reset()
        done = False
        while not done:
            action = targetPolicy[observation]
            observation_, reward, done, info = env.step(action)            
            observation = observation_
        totalReward += reward
        rewards[i] = totalReward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1
    
    wins /= numEpisodes
    losses /= numEpisodes
    draws /= numEpisodes
    print('win rate', wins, 'loss rate', losses, 'draw rate', draws)
    plt.plot(rewards)
    plt.show()