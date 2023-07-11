import sys
sys.path.append("..")
#import gymnasium as gym
import numpy as np
from envs.two_player_dubins_car import TwoPlayerDubinsCarEnv


import jax
import jax.numpy as jnp
import haiku as hk
import optax

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle 

def policy_network(observation):
    net = hk.Sequential([
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,

        hk.Linear(env.num_actions),
        jax.nn.softmax
    ])
    return net(observation)

policy_net = hk.without_apply_rng(hk.transform(policy_network))
key = jax.random.PRNGKey(42)
epsilon = 0.1

#Load data (deserialize)
with open('data/drone/test.pickle', 'rb') as handle:
     loaded_params = pickle.load(handle)

env = TwoPlayerDubinsCarEnv()
env.reset()
state = env.state


#


timesteps = np.arange(0, 10, 0.1)
for t in timesteps:
    if t % 0.5 == 0:
        prev_state = state
        for player in env.players:
            nn_state = env.encode_helper(state)
            action = env.select_action_sr(nn_state, loaded_params[player], policy_net,  key, epsilon)
            state, reward, done, info = env.step(state=state, action=action, player=player, update_env=True)
        pass

    attacker_prev_state = prev_state['attacker'][0:3]
    defender_prev_state = prev_state['defender'][0:3]

    attacker_state = state['attacker'][0:3]
    defender_state = state['defender'][0:3]

    lamb = (t % 0.5) * 2 
    attacker_pos = lamb * attacker_state + (1 - lamb) * attacker_prev_state  
    defender_pos = lamb * defender_state + (1 - lamb) * defender_prev_state

    defender_cf.cmdPosition((defender_pos[0], defender_pos[1], 1.))
    attacker_cf.cmdPosition((attacker_pos[0], attacker_pos[1], 1.))  
    time.sleep(0.1)
    # send position to crazyflie
    # wait for 0.1 seconds


