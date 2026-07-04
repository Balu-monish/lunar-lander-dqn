import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
      
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Layer 1: 8 inputs -> 64 hidden nodes
        self.fc1 = nn.Linear(state_size, 64)
        # Layer 2: 64 hidden nodes -> 64 hidden nodes
        self.fc2 = nn.Linear(64, 64)
        # Layer 3: 64 hidden nodes -> 4 outputs (one for each action)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state)) # Apply ReLU activation
        x = F.relu(self.fc2(x))     # Apply ReLU activation
        return self.fc3(x)          # Return the final Q-values