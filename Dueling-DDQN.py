
class DuelingDDQN(nn.Module):
    def __init__(self, input_shape, num_actions, name, lr=0.001):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)

        # here is the difference between dueling and not dueling dqn
        self.A = nn.Linear(512, self.num_actions)
        self.V = nn.Linear(512, 1)

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
        A = self.A(x)
        V = self.V(x)
        return V, A
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

'''
    the memory will be the same as before
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


class AgentDuelingDDQN:
    def __init__(self, input_shape, num_actions=4, lr=0.001, epsilon=1, epsilon_end=0.01, epsilon_decay=1e-6, gamma=0.99):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.Q_eval = DQN(input_shape, num_actions, 'eval', lr)
        self.Q_next = DQN(input_shape, num_actions, 'next', lr)
        self.memory = Memory(10000, input_shape)
        self.action_space = [x for x in range(num_actions)]
        self.step_learn = 0
        self.learn_traget = 1000
    
    def choose_action(self, obs):
        random = np.random.rand()
        if random <= self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            s = torch.Tensor(obs, dtype=float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(s)
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
        self.Q_eval.optim.zero_grad()
        s, a, r, s_, d = self.get_memory(batch_size)
        states = torch.tensor(s, dtype=float).to(self.Q_eval.device)
        actions = torch.tensor(a).to(self.Q_eval.device)
        rewards = torch.tensor(r).to(self.Q_eval.device)
        states_ = torch.tensor(s_, dtype=float).to(self.Q_eval.device)
        inds = [0,1,2,3]

        # different part (dueling)
        V_s, A_s = self.Q_eval.forward(states)
        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdims=True)))[inds, actions]
        V_s_, A_s_ = self.Q_next.forward(states_)
        q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdims=True)))
        V_s_pred, A_s_pred = self.Q_eval.forward(states_)
        q_eval_pred = torch.add(V_s_pred, (A_s_pred - A_s_pred.mean(dim=1, keepdims=True)))
        q_next[done] = 0
        max_a = torch.argmax(q_eval, dim=1) 
        q_traget = rewards + self.gamma*q_next[inds, max_a]
        agent_loss = self.Q_eval.loss(q_traget, q_pred)
        agent_loss.backward()
        self.Q_eval.optim.step()
        self.decay_eps()
        self.step_learn += 1

