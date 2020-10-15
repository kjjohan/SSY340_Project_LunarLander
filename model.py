import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyGradientNetwork(nn.module):
    def __init__(self, lr=0.0001, input_dims, layer1_dims, layer2_dims, n_actions)    
    #super(PolicyGradientNetwork, self).__init__()
    self.input_dims = input_dims
    self.layer1_dims = layer1_dims
    self.layer2_dims = layer2_dims
    self.n_actions = n_actions

    # Creating the network layers
    self.layer1 = nn.Linear(self.input_dims, self.layer1_dims)
    self.layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
    self.layer_output = nn.Linear(self.layer2_dims, self.n_actions)
    #self.optimizer = optim.Adam(self.parameters(), lr=lr)

    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')