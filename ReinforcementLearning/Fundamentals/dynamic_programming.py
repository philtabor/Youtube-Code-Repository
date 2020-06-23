import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):
    def __init__(self, m, n, magicSquares):
        self.grid = np.zeros((m,n))
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m*self.n)]
        self.stateSpace.remove(80)
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.possibleActions = ['U', 'D', 'L', 'R']
        self.actionSpace = {'U': -self.m, 'D': self.m,
                            'L': -1, 'R': 1}
        self.P = {}
        # dict with magic squares and resulting squares
        self.magicSquares = magicSquares
        self.initP()

    def initP(self):
        for state in self.stateSpace:
            for action in self.possibleActions:
                reward = -1
                state_ = state + self.actionSpace[action]
                if state_ in self.magicSquares.keys():
                    state_ = self.magicSquares[state_]
                if self.offGridMove(state_, state):
                    state_ = state
                if self.isTerminalState(state_):
                    reward = 0
                self.P[(state_, reward, state, action)] = 1

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def offGridMove(self, newState, oldState):
        # if we move into a row not in the grid
        if newState not in self.stateSpacePlus:
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.m == 0 and newState  % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False

def printV(V, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.m * idx + idy
            print('%.2f' % V[state], end='\t')
        print('\n')
    print('--------------------')

def printPolicy(policy, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.m * idx + idy
            if not grid.isTerminalState(state):
                if state not in grid.magicSquares.keys():
                    print('%s' % policy[state], end='\t')
                else:
                    print('%s' % '--', end='\t')
            else:
                print('%s' % '--', end='\t')
        print('\n')
    print('--------------------')

def evaluatePolicy(grid, V, policy, GAMMA, THETA):
    # policy evaluation for the random choice in gridworld
    converged = False
    i = 0
    while not converged:
        DELTA = 0
        for state in grid.stateSpace:
            i += 1
            oldV = V[state]
            total = 0
            weight = 1 / len(policy[state])
            for action in policy[state]:
                for key in grid.P:
                    (newState, reward, oldState, act) = key
                    # We're given state and action, want new state and reward
                    if oldState == state and act == action:
                        total += weight*grid.P[key]*(reward+GAMMA*V[newState])
            V[state] = total
            DELTA = max(DELTA, np.abs(oldV-V[state]))
            converged = True if DELTA < THETA else False
    print(i, 'sweeps of state space in policy evaluation')
    return V

def improvePolicy(grid, V, policy, GAMMA):
    stable = True
    newPolicy = {}
    i = 0
    for state in grid.stateSpace:
        i += 1
        oldActions = policy[state]
        value = []
        newAction = []
        for action in policy[state]:
            weight = 1 / len(policy[state])
            for key in grid.P:
                (newState, reward, oldState, act) = key
                # We're given state and action, want new state and reward
                if oldState == state and act == action:
                    value.append(np.round(weight*grid.P[key]*(reward+GAMMA*V[newState]), 2))
                    newAction.append(action)
        value = np.array(value)
        best = np.where(value == value.max())[0]
        bestActions = [newAction[item] for item in best]
        newPolicy[state] = bestActions

        if oldActions != bestActions:
            stable = False
    print(i, 'sweeps of state space in policy improvement')
    return stable, newPolicy

def iterateValues(grid, V, policy, GAMMA, THETA):
    converged = False
    i = 0
    while not converged:
        DELTA = 0
        for state in grid.stateSpace:
            i += 1
            oldV = V[state]
            newV = []
            for action in grid.actionSpace:
                for key in grid.P:
                    (newState, reward, oldState, act) = key
                    if state == oldState and action == act:
                        newV.append(grid.P[key]*(reward+GAMMA*V[newState]))
            newV = np.array(newV)
            bestV = np.where(newV == newV.max())[0]
            bestState = np.random.choice(bestV)
            V[state] = newV[bestState]
            DELTA = max(DELTA, np.abs(oldV-V[state]))
            converged = True if DELTA < THETA else False

    for state in grid.stateSpace:
        newValues = []
        actions = []
        i += 1
        for action in grid.actionSpace:
            for key in grid.P:
                (newState, reward, oldState, act) = key
                if state == oldState and action == act:
                    newValues.append(grid.P[key]*(reward+GAMMA*V[newState]))
            actions.append(action)
        newValues = np.array(newValues)
        bestActionIDX = np.where(newValues == newValues.max())[0]
        bestActions = actions[bestActionIDX[0]]
        policy[state] = bestActions
    print(i, 'sweeps of state space for value iteration')
    return V, policy

if __name__ == '__main__':
    # map magic squares to their connecting square
    magicSquares = {18: 54, 63: 14}
    env = GridWorld(9, 9, magicSquares)
    # model hyperparameters
    GAMMA = 1.0
    THETA = 1e-6 # convergence criteria

    V = {}
    for state in env.stateSpacePlus:
        V[state] = 0

    policy = {}
    for state in env.stateSpace:
        # equiprobable random strategy
        policy[state] = env.possibleActions

    V = evaluatePolicy(env, V, policy, GAMMA, THETA)
    printV(V, env)

    stable = False
    while not stable:
        V = evaluatePolicy(env, V, policy, GAMMA, THETA)

        stable, policy = improvePolicy(env, V, policy, GAMMA)

    printV(V, env)

    printPolicy(policy, env)

    # initialize V(s)
    V = {}
    for state in env.stateSpacePlus:
        V[state] = 0

    # Reinitialize policy
    policy = {}
    for state in env.stateSpace:
        policy[state] = [key for key in env.possibleActions]

    # 2 round of value iteration ftw
    for i in range(2):
        V, policy = iterateValues(env, V, policy, GAMMA, THETA)

    printV(V, env)
    printPolicy(policy, env)
