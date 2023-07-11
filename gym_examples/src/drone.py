import sys
sys.path.append("..")
#import gymnasium as gym
import numpy as np
from envs.two_player_dubins_car import TwoPlayerDubinsCarEnv
import copy


import jax
import jax.numpy as jnp
import haiku as hk
import optax

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle 

from crazyflie_py import Crazyswarm

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

def main():
    global state
    global key
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    defender_cf = swarm.allcfs.crazyflies[0]
    attacker_cf = swarm.allcfs.crazyflies[1]
    attacker_cf.setLEDColor(0, 0, 0)
    defender_cf.setLEDColor(0, 0, 0)
    defender_cf.takeoff(1.0, 2.5)
    attacker_cf.takeoff(1.0, 2.5)

    timeHelper.sleep(3.5)

    initial_position_d = (state['defender'][0], state['defender'][1], 1.)
    initial_position_a = (state['attacker'][0], state['attacker'][1], 1.)

    defender_cf.goTo(initial_position_d, 0, 3.5)
    attacker_cf.goTo(initial_position_a, 0, 3.5)
    timeHelper.sleep(5.)
    attacker_cf.setLEDColor(255, 0, 0)
    defender_cf.setLEDColor(0, 255, 0)
    
    timesteps = np.arange(0, 10, 0.1)
    for t in timesteps:
        if t % 0.5 == 0:
            prev_state = copy.deepcopy(state) 
            for player in env.players:
                key, subkey = jax.random.split(key)
                nn_state = env.encode_helper(state)
                action = env.select_action_sr(nn_state, loaded_params[player], policy_net,  subkey, epsilon)
                state, reward, done, info = env.step(state=state, action=action, player=player, update_env=True)
                print('prev_state', prev_state)
                print('state:', state)
                print('action:', action)


        attacker_prev_state = prev_state['attacker']
        defender_prev_state = prev_state['defender']

        attacker_state = state['attacker']
        defender_state = state['defender']

        lamb = (t % 0.5) * 2
        attacker_pos = lamb * attacker_state + (1 - lamb) * attacker_prev_state  
        defender_pos = lamb * defender_state + (1 - lamb) * defender_prev_state

        defender_cf.cmdPosition((defender_pos[0], defender_pos[1], 1.))
        attacker_cf.cmdPosition((attacker_pos[0], attacker_pos[1], 1.)) 


        timeHelper.sleepForRate(10)
        # send position to crazyflie
        # wait for 0.1 seconds
    defender_cf.notifySetpointsStop()
    attacker_cf.notifySetpointsStop()
    attacker_cf.setLEDColor(0, 0, 0)
    defender_cf.setLEDColor(0, 0, 0)
    pos = np.array(defender_cf.initialPosition) + np.array([0., 0., 1.0])
    defender_cf.goTo(pos, 0, 5.0)
    pos = np.array(attacker_cf.initialPosition) + np.array([0., 0., 1.0])
    attacker_cf.goTo(pos, 0, 5.0)
    timeHelper.sleep(6.0)

    defender_cf.land(0.04, 2.5)
    attacker_cf.land(0.04,2.5)

    timeHelper.sleep(5.0)

    

if __name__ == '__main__':
    main()
    #ros2 launch crazyflie launch.py backend:=sim
    #adjust crazyf;ies.yaml



