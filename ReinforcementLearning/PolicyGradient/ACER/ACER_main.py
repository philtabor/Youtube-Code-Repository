import gym
import numpy as np
from ACER import Agent
import matplotlib.pyplot as plt

if __name__== '__main__':
    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 32
    alpha = 0.0003
    agent = Agent(lr = alpha, input_dims = env.observation_space.shape[0], action_dims = env.action_space.n)

    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_games = 100000

    for i in range(n_games):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state,reward,done,info = env.step(action)
            agent.memory.append((state,reward,next_state,done))
            state = next_state
            score += reward
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        agent.learn()
        if avg_score>best_score:
            best_score = avg_score
            agent.save_model()
        if i%N == 0:
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.critic_target.load_state_dict(agent.critic.state_dict())

        print(f'episode {i} score: {score} ave_score: {avg_score} ')
    
    plt.plot(score_history)

        


    
        