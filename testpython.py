import gym

env = gym.make('LunarLander-v2')
env.reset()
terminate = False
#for _ in range(1000):
while not terminate:
    env.render()
    _, _, terminate, _ = env.step(env.action_space.sample())
    
env.close()