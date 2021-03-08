# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:54:58 2021

@author: ggpen
"""

import gym, math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd 

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class NoisyDuelingDDQN(nn.Module):
    def __init__(self, input_shape, num_actions, name,  lr=0.001):
        super(NoisyDuelingDDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.noisy1 = NoisyLinear(self.feature_size(), 512)

        # here is the difference between dueling and not dueling dqn
        self.A = nn.Linear(512, self.num_actions)
        self.V = nn.Linear(512, 1)
        
        
        self.optim = optim.RMSprop(self.parameters(),lr = lr)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.name = name
        
    def forward(self, x):
        x = x.view(-1,1,185,95)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.noisy1(x))
        A = self.A(x)
        V = self.V(x)
        return V, A
    
    def reset_noise(self):
        self.noisy1.reset_noise()
        
    
    def update(self, q_traget, q_pred, importance,eps):
        self.q_traget = q_traget
        self.q_pred = q_pred
        self.importance = importance
        loss, error = self.update_loss(importance, eps)
        return loss, error
    
    def update_loss(self, importance,eps):
        error = self.q_traget- self.q_pred
        loss = torch.mean(torch.multiply(torch.square(error), torch.tensor(importance*(1-eps)).to(self.device)))
        return loss, error
    
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
        self.state_memory = np.zeros((mem_size, *in_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((mem_size, *in_dims), dtype=np.float32)
        self.action_memory = np.zeros((mem_size), dtype=np.int64)
        self.reward_memory = np.zeros((mem_size), dtype=np.int64)
        self.done_memory = np.zeros((mem_size), dtype=np.uint8)
        # we add new storage for prioritized frames
        self.prior_memory = np.zeros((mem_size), dtype=np.float32)

    def store_memory(self, state, action, reward, state_, done):
        indx = self.mem_counter%self.mem_size
        self.state_memory[indx] = state
        self.next_state_memory[indx] = state_
        self.action_memory[indx] = action
        self.reward_memory[indx] = reward
        self.done_memory[indx] = done
        # new
        self.prior_memory[indx] = 1 if max(self.prior_memory)==0 else max(self.prior_memory)
        self.mem_counter += 1
    
    #new
    def get_probs(self, power):
        scaled_priors = self.prior_memory**power
        sample_probs = scaled_priors/ sum(scaled_priors)
        return sample_probs

    def get_importance(self, probs):
        importance = 1/self.mem_size * 1/probs
        importance = importance/ max(importance)
        return importance

    def sample_memory(self, batch_size, power=1):
        if self.mem_counter >= batch_size:
            # new
            probs = self.get_probs(power)
            if self.mem_counter >= self.mem_size:
                probs = probs[:self.mem_size]
                indx = np.random.choice(self.mem_size, batch_size, replace=False, p=probs)
            else:
                probs = probs[:self.mem_counter]
                indx = np.random.choice(self.mem_counter, batch_size, replace=False, p=probs)
            state = self.state_memory[indx]
            action = self.action_memory[indx]
            reward = self.reward_memory[indx]
            state_ = self.next_state_memory[indx]
            done = self.done_memory[indx]
            importance = self.get_importance(probs[indx])
            return state, action, reward, state_, done, importance, indx
    
    def add_prior(self, error, indx):
        for i,e in zip(indx, error):
            self.prior_memory[i] = abs(e)

class NoisyPrioritizedAgentDuelingDDQN:
    def __init__(self, input_shape, num_actions=4, lr=0.001, epsilon=1, epsilon_end=0.01, epsilon_decay=1e-6, gamma=0.99):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.Q_eval = NoisyDuelingDDQN(input_shape, num_actions, 'eval', lr)
        self.Q_next = NoisyDuelingDDQN(input_shape, num_actions, 'next', lr)
        self.memory = Memory(10000, input_shape)
        self.action_space = [x for x in range(num_actions)]
        self.step_learn = 0
        self.learn_traget = 1000
    
    def choose_action(self, obs):
        s = torch.tensor(obs, dtype=torch.float).to(self.Q_eval.device)
        _, actions = self.Q_eval.forward(s)
        action = torch.argmax(actions).item()
        return action
    
    def decay_eps(self):
        self.epsilon = (self.epsilon - self.epsilon_decay) if self.epsilon_end < self.epsilon else self.epsilon_end
    
    def write_memory(self, s, a, r, s_, d):
        self.memory.store_memory(s, a, r, s_, d)
    
    def get_memory(self, batch_size):
        return self.memory.sample_memory(batch_size)
    
    def switch_learner(self):
        if self.step_learn % self.learn_traget == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def learn(self, batch_size=4):
        if self.memory.mem_counter > 4:
            self.Q_eval.optim.zero_grad()
            # new
            s, a, r, s_, d, importance, indx = self.get_memory(batch_size)
            states = torch.tensor(s, dtype=torch.float).to(self.Q_eval.device)
            actions = torch.tensor(a).to(self.Q_eval.device)
            rewards = torch.tensor(r).to(self.Q_eval.device)
            states_ = torch.tensor(s_, dtype=torch.float).to(self.Q_eval.device)
            inds = [0,1,2,3]
    
            # different part (dueling)
            V_s, A_s = self.Q_eval.forward(states)
            q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdims=True)))[inds, actions]
            V_s_, A_s_ = self.Q_next.forward(states_)
            q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdims=True)))
            V_s_pred, A_s_pred = self.Q_eval.forward(states_)
            q_eval_pred = torch.add(V_s_pred, (A_s_pred - A_s_pred.mean(dim=1, keepdims=True)))
            q_next[done] = 0
            max_a = torch.argmax(q_eval_pred, dim=1) 
            q_traget = rewards + self.gamma*q_next[inds, max_a]
            agent_loss, error = self.Q_eval.update(q_traget, q_pred, importance, self.epsilon)
            agent_loss.backward()
            agent.memory.add_prior(error, indx)
            self.Q_eval.optim.step()
            self.decay_eps()
            self.step_learn += 1
            self.Q_eval.reset_noise()
            self.Q_next.reset_noise()    
            return agent_loss
    