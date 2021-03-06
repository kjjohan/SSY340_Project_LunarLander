import numpy as np
import gym
from modelA2C import PolicyGradientAgent
import matplotlib.pyplot as plt 
from gym import wrappers
import time

def plotLearning(scores, filename, x=None, window=25):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.rc('font', family='serif')
    plt.ylabel('Score')       
    plt.xlabel('Game')            
    plt.title('A2C')         
    plt.plot(x, scores, alpha=0.7)
    plt.plot(x, running_avg, linewidth=3.0)
    plt.ylim((-500,300))
    plt.legend(['Score', 'Smoothed score over 25 episodes'], loc='lower right')
    
    plt.savefig(filename+'.eps', format='eps', dpi=1000)
    plt.savefig(filename+'.png')


if __name__ == '__main__':
    agent = PolicyGradientAgent(actor_lr=0.002, critic_lr=0.002, input_dims=[8], actor_dims=[64,32], critic_dims=[64,32], n_actions=4, discount_rate=0.99)

    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episodes = 3001
    env = wrappers.Monitor(env, "gifsA2C002", video_callable=lambda count: count % 500 == 0, force=True)

    start_time = time.time()
    for i in range(num_episodes):
        print('episode: ', i, 'score: ', score)
        done = False
        score = 0
        state = env.reset() 

        while not done:
            probabilities = agent.actor_network.forward(state)
            value = agent.critic_network.forward(state)

            action = agent.choose_action(probabilities)
            state_, reward, done, info = env.step(action)

            agent.store_values(value)
            agent.store_rewards(reward)

            state = state_
            score += reward

        score_history.append(score)
        agent.learn()

    elapsed_time = time.time() - start_time
    print("Elapsed time: ", elapsed_time)
    filename = 'images/A2C-alpha002'
    plotLearning(score_history, filename=filename, window=25)




    
