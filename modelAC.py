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
        self.actor_layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        #self.actor_layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.actor_layer_output = nn.Linear(self.layer1_dims, self.n_actions)

        #self.critic_layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        #self.critic_layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.critic_layer_output = nn.Linear(self.layer1_dims, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate, betas=betas)

    def forward(self, observation):
        state = T.Tensor(observation)
        state = self.actor_layer1(state)
        state = F.relu(state)
        
        #probs = self.actor_layer1(state)
        #probs = F.relu(probs)
        #probs = self.actor_layer2(state)
        #probs = F.relu(probs)
        probs = self.actor_layer_output(state)
        probs = F.softmax(probs, dim=0)

        #value = self.critic_layer1(state)
        #value = F.relu(value)
        #value = self.critic_layer2(state)
        #value = F.relu(value)
        value = self.critic_layer_output(state)

        return probs, value

class PolicyGradientAgent(object):
    def __init__(self, learn_rate, input_dims, layer1_dims=256, layer2_dims=256, n_actions=4, discount_rate=0.99, betas=(0.9, 0.999)):   
        self.discount_rate = discount_rate
        self.reward_memory = []
        self.value_memory = []
        self.action_memory = []
        self.network = PolicyGradientNetwork(learn_rate, input_dims, layer1_dims, layer2_dims, n_actions, betas)

    def choose_action(self, probabilities):
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def store_values(self, value):
        self.value_memory.append(value)

    def learn(self):

        self.network.optimizer.zero_grad()
        Q_values = np.zeros_like(self.reward_memory, dtype=np.float64)
        Q_value = 0
        for t in reversed(range(len(self.reward_memory))):
            Q_value = self.reward_memory[t] + self.discount_rate*Q_value
            Q_values[t] = Q_value

        mean = np.mean(Q_values)
        std = np.std(Q_values) if np.std(Q_values) > 0 else 1
        Q_values = (Q_values - mean) / std

        #self.value_memory = np.array(self.value_memory, dtype=np.float64)
        #mean = np.mean(self.value_memory)
        #std = np.std(self.value_memory) if np.std(self.value_memory) > 0 else 1
        #self.value_memory = (self.value_memory - mean) / std

        Q_values = T.tensor(Q_values, dtype=T.float)
        self.value_memory = T.tensor(self.value_memory, dtype=T.float)

        actor_loss = 0
        critic_loss = 0
        AC_loss = 0
        
        for q, logprob, value in zip(Q_values, self.action_memory, self.value_memory):
            advantage = q - value.detach()
            actor_loss += -advantage * logprob
            critic_loss += F.smooth_l1_loss(value.detach(), q)
            AC_loss += actor_loss + critic_loss

        AC_loss.backward()
        self.network.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
        self.value_memory = []


