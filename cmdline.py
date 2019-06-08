import argparse
import gym
from ReinforcementLearning.DeepQLearning.utils import plotLearning
from ReinforcementLearning.DeepQLearning.simple_dqn_torch import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    description='Command line Utility for training RL models')
    # the hyphen makes the argument optional
    parser.add_argument('-n_games', type=int, default=1,
                        help='Number of games to play')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('-eps_end', type=float, default=0.01,
            help='Final value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-gamma', type=float, default=0.99,
                                    help='Discount factor for update equation.')
    parser.add_argument('-env', type=str, default='LunarLander-v2',
                                        help='OpenAI gym environment for agent')
    parser.add_argument('-eps_dec', type=float, default=0.996,
                        help='Multiplicative factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=1.0,
        help='Starting value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-max_mem', type=int, default=1000000,
                                help='Maximum size for memory replay buffer')
    parser.add_argument('-dims', type=int, default=8,
                            help='Input dimensions; matches env observation, \
                                  must be list or tuple')
    parser.add_argument('-bs', type=int, default=32,
                            help='Batch size for replay memory sampling')
    parser.add_argument('-n_actions', type=int, default=4,
                            help='Number of actions in discrete action space')
    args = parser.parse_args()

    env = gym.make(args.env)

    args.dims = [args.dims]

    agent = Agent(args.gamma, args.eps, args.lr, args.dims, args.bs,
                  args.n_actions, args.max_mem, args.eps_end, args.eps_dec)

    eps_history, scores = [], []
    for i in range(args.n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.chooseAction(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.storeTransition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
            agent.learn()

        eps_history.append(agent.EPSILON)
        scores.append(score)

        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % agent.EPSILON)
        else:
            print('episode: ', i,'score: ', score)

    x = [i+1 for i in range(args.n_games)]
    # filename should reflect whatever it is you are varying to tune your
    # agent. For simplicity I'm just showing alpha and gamma, but it can be
    # the epsilons as well. You can even include parameters for the fully
    # connected layers and use them as part of the file name.
    filename = args.env + '_alpha' + str(args.lr) + '_gamma' + str(args.gamma) + \
              '.png'
    plotLearning(x, scores, eps_history, filename)
