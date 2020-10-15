import numpy as np
import gym
from model import PolicyGradientAgent
#import matplotlib.pyplot as plt
#from utils import plotLearning
#from gym import wrappers

if __name__ == '__main__':
    agent = PolicyGradientAgent(learn_rate=0.0001, input_dims=[8], layer1_dims=128, layer2_dims=128, n_actions=4, discount_rate=0.99)
    print('Ja det funkar!!!!!!!!!!!!!')
    print(agent)