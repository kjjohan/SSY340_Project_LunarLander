import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyGradientNetwork(nn.Module):
    def __init__(self, learn_rate, input_dims, layer1_dims, layer2_dims, n_actions, betas):    
        super(PolicyGradientNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.n_actions = n_actions

        # Creating the network layers
        self.layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        #self.layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.layer_output = nn.Linear(self.layer1_dims, self.n_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate, betas=betas)

    def forward(self, x):
        state = T.Tensor(x)
        x = self.layer1(state)
        x = F.relu(x)
        #x = self.layer2(x)
        #x = F.relu(x)
        x = self.layer_output(x)
        x = F.softmax(x, dim=0)
        return x

class PolicyGradientAgent(object):
    def __init__(self, learn_rate, input_dims, layer1_dims=256, layer2_dims=256, n_actions=4, discount_rate=0.99, betas=(0.9,0.999)):   
        self.discount_rate = discount_rate
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyGradientNetwork(learn_rate, input_dims, layer1_dims, layer2_dims, n_actions, betas)

    def choose_action(self, probabilities):
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k]*discount
                discount *= self.discount_rate
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        G = T.tensor(G, dtype=T.float)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []


