import gymnasium as gym
env = gym.make('CartPole-v1')
env.reset()

for i in range(1000):
	print(i)
	env.step(env.action_space.sample())
	env.render()
