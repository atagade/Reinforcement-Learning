import gym
import random
import numpy as np

epsilon = 1
learning_rate = 0.8
gamma = 0.9
training_eps = 10000
rewards = []

env = gym.make("FrozenLake-v0", is_slippery=False)
qtable = np.zeros((env.observation_space.n,env.action_space.n))

for episode in range(training_eps):
	total_reward = 0
	state = env.reset()

	for i in range(100):
		exp_exp_tradeoff = random.uniform(0,1)
		if(exp_exp_tradeoff<epsilon):
			action = env.action_space.sample()
		else:
			action = np.argmax(qtable[state,:])

		next_state, reward, done, info = env.step(action)
		qtable[state, action] = qtable[state, action] + learning_rate*(reward + gamma * np.max(qtable[next_state,:]) - qtable[state,action])
		#print(qtable[state,action])
		state = next_state
		total_reward += reward
		rewards.append(total_reward)
		epsilon = 0.1 + (0.99 * np.exp(-0.005*episode))
		if(done):
			break

print('Average reward = '+str(sum(rewards)/training_eps))
print(qtable)

for episode in range(5):
	state = env.reset()
	total_reward = 0
	
	for step in range(100):
		action = np.argmax(qtable[state,:])
		next_state, reward, done, info = env.step(action)
		total_reward += reward
		if(done):
			print('Steps=',str(step+1))
			print('Reward=',str(total_reward))
			env.render()
			break
		state = next_state
		