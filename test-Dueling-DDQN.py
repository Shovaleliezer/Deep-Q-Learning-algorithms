'''
    here we will test our dqn agent on space-invaders
    first install gym with pip install gym
    and import our agent from Dueling-DDQN
    not really much to change.
'''

import gym
import torch
import numpy as np
from Dueling-DDQN import AgentDuelingDDQN 
NUM_OF_GAMES = 10000
env = gym.make('Space-Invaders-v0')
INPUT_SHAPE = (185,95)
# the only change
agent = AgentDuelingDDQN()
for episode in range(NUM_OF_GAMES):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = agnet.choose_action(state)
        state_, reward, done, info = env.step(action)
        agent.write_memory(np.mean(state[15:200,30:125], axis=2), action, reward, np.mean(state_[15:200,30:125]), done)
        agent.learn()
        score += reward
        state = state_
    print(f'episode: {episode}, score:{score}, epsilon:{agent.epsilon}')