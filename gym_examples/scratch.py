#import gymnasium as gym

import sys
import numpy as np
import gym_examples
import gymnasium as gym

from src.value_iteration import ValueIteration

from gym_examples.envs.grid_world import GridWorldEnv



# Create the environment
#env = gym.make("FrozenLake-v1")
env = gym.make('gym_examples/FrozenLake-v2')

# Create the value iteration object
vi = ValueIteration(env)

# Get the policy
policy = vi.run()
# Use the policy to interact with the environment
total_rewards = 0
num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()[0]
    terminated = False
    while not terminated:
       #action = env.action_space.sample()
        action = np.argmax(policy[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward
        state = next_state


# Print the average reward
print(f"Average reward: {total_rewards / num_episodes}")
