import sys
sys.path.append("..")
#import gymnasium as gym
import numpy as np
from envs.two_player_dubins_car import TwoPlayerDubinsCarEnv
import copy


import jax
import jax.numpy as jnp
import haiku as hk
#import optax

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle 
import yaml
import datetime


from crazyflie_py import Crazyswarm


def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def print_config(config):
    print("Starting experiment with the following configuration:\n")
    print(yaml.dump(config))


def policy_network(observation, legal_moves):
        net = hk.Sequential(
            [
                hk.Linear(64),
                jax.nn.relu,
                hk.Linear(64),
                jax.nn.relu,
                hk.Linear(64),
                jax.nn.relu,
                hk.Linear(64),
                jax.nn.relu,
                hk.Linear(env.num_actions),
                jax.nn.softmax,
            ]
        )
        # legal_moves = legal_moves[..., None]

        logits = net(observation)

        #masked_logits = jnp.where(legal_moves, logits, -1e8)
        masked_logits = jnp.where(legal_moves, logits, 1e-8)

policy_net = hk.without_apply_rng(hk.transform(policy_network))
key = jax.random.PRNGKey(1)
epsilon = 0.1

#Load data (deserialize)
with open('data/drone/stackelberg_camera.pickle', 'rb') as handle:
     loaded_params = pickle.load(handle)


config = load_config("configs/config.yml")
print_config(config)

game_type = config['game']['type']
timestamp = str(datetime.datetime.now())

env = TwoPlayerDubinsCarEnv(
    game_type=game_type,
    num_actions=config['env']['num_actions'],
    size=config['env']['board_size'],
    reward=config['env']['reward'],
    max_steps=config['env']['max_steps'],
    init_defender_position=config['env']['init_defender_position'],
    init_attacker_position=config['env']['init_attacker_position'],
    capture_radius=config['env']['capture_radius'],
    goal_position=config['env']['goal_position'],
    goal_radius=config['env']['goal_radius'],
    timestep=config['env']['timestep'],
    v_max=config['env']['velocity'],
    omega_max=config['env']['turning_angle'],
)

key = jax.random.PRNGKey(1)

env.reset(key)
state = env.state
real = True

#

def main():
    global state
    global key
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    defender_cf = swarm.allcfs.crazyflies[0]
    attacker_cf = swarm.allcfs.crazyflies[1]

    if real:
        attacker_cf.setLEDColor(0, 0, 0)
        defender_cf.setLEDColor(0, 0, 0)
    defender_cf.takeoff(1.0, 2.5)
    attacker_cf.takeoff(1.0, 2.5)

    timeHelper.sleep(3.5)

    initial_position_d = (state['defender'][0]/2, state['defender'][1]/2, 1.)
    initial_position_a = (state['attacker'][0]/2, state['attacker'][1]/2, 1.)

    defender_cf.goTo(initial_position_d, 0, 3.5)
    attacker_cf.goTo(initial_position_a, 0, 3.5)
    timeHelper.sleep(5.)
    if real:
        attacker_cf.setLEDColor(255, 0, 0)
        defender_cf.setLEDColor(0, 0, 255)
        
    timesteps = np.arange(0, 30, 0.1)
    for t in timesteps:
        if t % 0.5 == 0:
            prev_state = copy.deepcopy(state) 
            for player in env.players:
                key, subkey = jax.random.split(key)
                legal_actions_mask = env.get_legal_actions_mask(state, player)
                nn_state = env.encode_helper(state)
                action = env.constrained_select_action(nn_state, policy_net, loaded_params[player], legal_actions_mask, subkey, 0.0001)
                state, reward, done, info = env.step(state=state, action=action, player=player, update_env=True)
                env.render()
                print('prev_state', prev_state)
                print('state:', state)
                print('action:', action)

        if done: 
            if real:
                if info['status'] == 'goal_reached':
                    attacker_cf.setLEDColor(255, 255, 0)
                    defender_cf.setLEDColor(255, 255, 0)
                elif info['status'] == 'eaten':
                    attacker_cf.setLEDColor(0, 255, 255)
                    defender_cf.setLEDColor(0, 255, 255)
            break
            

        attacker_prev_state = prev_state['attacker']
        defender_prev_state = prev_state['defender']

        attacker_state = state['attacker']
        defender_state = state['defender']

        lamb = (t % 0.5) * 2
        attacker_pos = lamb * attacker_state + (1 - lamb) * attacker_prev_state  
        defender_pos = lamb * defender_state + (1 - lamb) * defender_prev_state

        defender_cf.cmdPosition((defender_pos[0]/2, defender_pos[1]/2, 1.))
        attacker_cf.cmdPosition((attacker_pos[0]/2, attacker_pos[1]/2, 1.)) 


        timeHelper.sleepForRate(20)
        # send position to crazyflie
        # wait for 0.1 seconds
    defender_cf.notifySetpointsStop()
    attacker_cf.notifySetpointsStop()
    if real:
        attacker_cf.setLEDColor(0, 0, 0)
        defender_cf.setLEDColor(0, 0, 0)
    pos = np.array(defender_cf.initialPosition) + np.array([0., 0., 1.0])
    defender_cf.land(0.04, 2.5)
    pos = np.array(attacker_cf.initialPosition) + np.array([0., 0., 1.0])
    attacker_cf.land(0.04,2.5)
    timeHelper.sleep(6.0)


    timeHelper.sleep(5.0)

    env.make_gif('gifs/lab/test_real.gif')
    

if __name__ == '__main__':
    main()
    #ros2 launch crazyflie launch.py backend:=sim
    #adjust crazyfies.yaml



