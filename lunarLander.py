import numpy as np
import gym
from model import PolicyGradientAgent
import matplotlib.pyplot as plt 
from gym import wrappers

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

if __name__ == '__main__':
    agent = PolicyGradientAgent(learn_rate=0.0005, input_dims=[8], layer1_dims=256, layer2_dims=256, n_actions=4, discount_rate=0.99)
    #agent.load_checkpoint()

    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episodes = 1000
    env = wrappers.Monitor(env, "gifs/lunar-lander", video_callable=lambda count: count % 200 == 0, force=True)

    for i in range(num_episodes):
        print('episode: ', i, 'score: ', score)
        done = False
        score = 0
        observation = env.reset() 
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_rewards(reward)
            observation = observation_
            score += reward
        score_history.append(score)
        agent.learn()
        #agent.save_checkpoint()
    filename = 'lunar-lander-alpha0005-256x256-discount099-e1000.png'

    plotLearning(score_history, filename=filename, window=15)




    
