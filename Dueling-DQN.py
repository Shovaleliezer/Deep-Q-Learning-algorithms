
class DuelingDDQN(nn.Module):
    def __init__(self, input_shape, num_actions, name, lr=0.001):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)
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
