from audioop import avg
import gym
import numpy as np
from ACER_online import Agent
import matplotlib.pyplot as plt
import os

if __name__== '__main__':
    print(os.system('pwd'))
    env = gym.make('CartPole-v1')
    batch_size = 32
    alpha = 0.0003
    agent = Agent(lr = alpha, input_dims = env.observation_space.shape[0], action_dims = env.action_space.n)

    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_games = 2000

    for i in range(n_games):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state,reward,done,info = env.step(action)
            agent.memory.append((state,action,reward,next_state,done))
            # agent.learn()
            state = next_state
            score += reward
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        agent.learn()
        if avg_score>best_score:
            best_score = avg_score
            agent.save_model()
        agent.memory.clear()
        print(f'episode {i} score: {score} ave_score: {avg_score} memory_buffer {len(agent.memory)}')
        if avg_score>=500:
            break
    
    plt.plot(score_history)
    agent.load_model()
    done = False
    state = env.reset()
    while True:
        action= agent.choose_action(state)
        state, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()

        


    
        