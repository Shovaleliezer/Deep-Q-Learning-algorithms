'''
    here we will test our dqn agent on space-invaders
    first install gym with pip install gym
    and import our agent from DQN
'''

import gym
import torch
import numpy as np
from DQN import AgentDQN 
NUM_OF_GAMES = 10000
env = gym.make('Space-Invaders-v0')
INPUT_SHAPE = (185,95)
agent = AgentDQN()
for episode in range(NUM_OF_GAMES):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = agnet.choose_action(state)
        state_, reward, done, info = env.step(action)
        agent.memory.store_memory(state, action, reward, state_, done)
        agent.learn()
        state = state_
        score += score
    print(f'episode: {episode}, score:{score}, epsilon:{agent.epsilon}')