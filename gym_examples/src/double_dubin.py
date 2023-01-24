import gym
from gym import spaces
import numpy as np
from gym_examples.envs.dubins_car import TwoPlayerDubinsCarEnv


import jax
import jax.numpy as jnp
import haiku as hk
import optax


#generate data

env = TwoPlayerDubinsCarEnv()
state = env.reset()
X = []
y = []
state = env.reset()
for i in range(100):
    for agent in env.players:

        action = env.action_space[agent].sample() 
        state, reward, done, info = env.step(action, agent)
        X.append(state)
        y.append(reward)
        env.render()
env.make_gif()

