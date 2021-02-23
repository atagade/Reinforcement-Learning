import gym
from gym.wrappers.monitoring import video_recorder
num_episodes = 1
num_timesteps = 5000

env = gym.make('Tennis-v0')
#vid = video_recorder.VideoRecorder(env, path='recording/training.mp4')
env = gym.wrappers.Monitor(env, 'recording', force=True)
for i in range(num_episodes):

	state = env.reset()
	total_reward = 0

	for j in range(num_timesteps):

		env.render()
		#vid.capture_frame()
		random_action = env.action_space.sample()
		next_state, reward, done, info = env.step(random_action)
		total_reward += reward
		if done:
			break
		if j%10==0:
			print('Episode ' + str(i+1) + ': Timestep '+str(j+1)+' Total Reward = '+str(total_reward))
	env.close()