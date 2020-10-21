import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, learn_rate, input_dims, layer1_dims, layer2_dims, n_actions):    
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.n_actions = n_actions

        # Creating the network layers
        self.actor_layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        self.actor_layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.actor_layer_output = nn.Linear(self.layer2_dims, self.n_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    def forward(self, state):
        state = T.Tensor(state)
        
        probs = self.actor_layer1(state)
        probs = F.relu(probs)
        probs = self.actor_layer2(probs)
        probs = F.relu(probs)
        probs = self.actor_layer_output(probs)
        probs = F.softmax(probs, dim=0)

        return probs

class CriticNetwork(nn.Module):
    def __init__(self, learn_rate, input_dims, layer1_dims, layer2_dims, n_actions):    
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.n_actions = n_actions

        # Creating the network layers
        self.critic_layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        self.critic_layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.critic_layer_output = nn.Linear(self.layer2_dims, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    def forward(self, state):
        state = T.Tensor(state)
        
        value = self.critic_layer1(state)
        value = F.relu(value)
        value = self.critic_layer2(value)
        value = F.relu(value)
        value = self.critic_layer_output(value)

        return value
        

class PolicyGradientAgent(object):
    def __init__(self, actor_lr, critic_lr, input_dims, actor_dims=[128, 128], critic_dims=[128, 128], n_actions=4, discount_rate=0.99):   
        self.discount_rate = discount_rate
        self.reward_memory = []
        self.value_memory = []
        self.action_memory = []
        self.actor_network = ActorNetwork(actor_lr, input_dims, actor_dims[0], actor_dims[1], n_actions)
        self.critic_network = CriticNetwork(critic_lr, input_dims, critic_dims[0], critic_dims[1], n_actions)

    def choose_action(self, probs):
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def store_values(self, value):
        self.value_memory.append(value)

    def learn(self):

        # Calculate R-values from terminal state and back to start
        R = np.zeros_like(self.reward_memory, dtype=np.float64)
        R_value = 0
        memory_length = len(self.reward_memory)
        for t in reversed(range(memory_length)):
            R_value = self.reward_memory[t] + self.discount_rate*R_value
            R[t] = R_value

        # Normalize
        mean = np.mean(R)
        std = np.std(R) if np.std(R) > 0 else 1
        R = (R - mean) / std

        # Convert to tensor
        R = T.tensor(R, dtype=T.float)
        self.value_memory = T.tensor(self.value_memory, dtype=T.float)

        # Calculate loss
        actor_loss = 0
        critic_loss = 0
        for r, logprob, value in zip(R, self.action_memory, self.value_memory):
            advantage = r - value
            actor_loss += -advantage * logprob
            critic_loss += F.smooth_l1_loss(value, r)
                
        # Update actor parameters
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward()
        self.actor_network.optimizer.step()

        # Update critic parameters
        self.critic_network.optimizer.zero_grad()
        critic_loss.requires_grad = True
        critic_loss.backward()
        self.critic_network.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
        self.value_memory = []


