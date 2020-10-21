import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyGradientNetwork(nn.Module):
    def __init__(self, learn_rate, input_dims, layer1_dims, layer2_dims, n_actions):    
        super(PolicyGradientNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.n_actions = n_actions

        # Creating the network layers
        self.layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        self.layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.layer_output = nn.Linear(self.layer2_dims, self.n_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    def forward(self, state):
        state = T.Tensor(state)
        probs = self.layer1(state)
        probs = F.relu(probs)
        probs = self.layer2(probs)
        probs = F.relu(probs)
        probs = self.layer_output(probs)
        probs = F.softmax(probs, dim=0)
        return probs

class PolicyGradientAgent(object):
    def __init__(self, learn_rate, input_dims, layer1_dims=128, layer2_dims=128, n_actions=4, discount_rate=0.99):   
        self.discount_rate = discount_rate
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyGradientNetwork(learn_rate, input_dims, layer1_dims, layer2_dims, n_actions)

    def choose_action(self, probs):
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        
        # Calculate R-values from terminal state and back to start
        R = np.zeros_like(self.reward_memory, dtype=np.float64)
        R_value = 0
        memory_length = len(self.reward_memory)
        for t in reversed(range(memory_length)):
            R_value = self.reward_memory[t] + self.discount_rate*R_value
            R[t] = R_value

        # Convert to tensor
        R = T.tensor(R, dtype=T.float)

        # Calculate loss
        loss = 0
        for r, logprob in zip(R, self.action_memory):
            loss += -r * logprob

        # Update parameters
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []


