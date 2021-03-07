# this code will show the simple dqn model
import torch.nn as nn
import torch 
import torch.optim as optim
from torch.nn import MSELoss
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, lr=0.001):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions)
        self.loss = MSELoss()
        self.optim = optim.RMSprop(self.parameters(),lr = lr)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.name = name
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

'''
    so why do we need memory for our agent?
    of course the answer for this is because we cant know what happends in the game by just one frame we need more frames.
    if we take batch_size number of frames.
    for example 4 so we can see the direction of the agent and the envirment,
    whice can help figure out what is the best action to do
'''
class Memory:
    def __init__(self, mem_size, in_dims):
        self.mem_size = mem_size
        self.mem_counter = 0
        self.state_memory = np.zeros((mem_size, in_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((mem_size, in_dims), dtype=np.float32)
        self.action_memory = np.zeros((mem_size), dtype=np.int64)
        self.reward_memory = np.zeros((mem_size), dtype=np.int64)
        self.done_memory = np.zeros((mem_size), dtype=np.uint8)

    def store_memory(self, state, action, reward, state_, done):
        indx = self.mem_counter%self.mem_size
        self.state_memory[indx] = state
        self.next_state_memory[indx] = state_
        self.action_memory[indx] = action
        self.reward_memory[indx] = reward
        self.done_memory[indx] = done
    
    def sample_memory(self, batch_size):
        if self.mem_counter >= batch_size:
            if self.mem_counter >= self.mem_size:
                indx = np.random.choice(self.mem_size, batch_size, replace=False)
            else:
                indx = np.random.choice(self.mem_counter, batch_size, replace=False)
            state = self.state_memory[indx]
            action = self.action_memory[indx]
            reward = self.reward_memory[indx]
            state_ = self.next_state_memory[indx]
            done = self.done_memory[indx]
            return state, aciton, reward, state_, done
