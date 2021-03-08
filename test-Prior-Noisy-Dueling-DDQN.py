    
 
import gym
import torch
import numpy as np
from Prior-Noisy-Dueling-DDQN import NoisyPrioritizedAgentDuelingDDQN    


env = gym.make('SpaceInvaders-v0')
agent = NoisyPrioritizedAgentDuelingDDQN((1,185,95), 6)
for episode in range(1000):
    done = False
    s = env.reset()
    score= 0
    loss = 0 
    while not done:
        env.render()
        a = agent.choose_action(np.mean(s[15:200,30:125],axis=2))
        state_, reward, done, info = env.step(a)
        if done and info['ale.lives'] == 0:
            reward = -100
        agent.write_memory(np.mean(s[15:200,30:125], axis=2), a, reward, np.mean(state_[15:200,30:125]), done)
        agent_loss = agent.learn()
        score += reward
        s = state_
        
    print(f'score is: {score}, episode: {episode}, epsilon: {agent.epsilon:.2f}')
env.close()

    
        
        
        
        
        
        