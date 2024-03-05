import sys
sys.path.append("..")
#import gymnasium as gym
import numpy as np
from envs.two_player_dubins_car_jax import TwoPlayerDubinsCarEnv
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


env = TwoPlayerDubinsCarEnv(
        game_type='stackelberg',
        num_actions=3,
        size=3,
        reward='',
        max_steps=50,
        capture_radius=0.5,
        goal_position=[0,-3],
        goal_radius=1,
        timestep=1,
        v_max=0.30,
        omega_max=30,
    )

def print_config(config):
    print("Starting experiment with the following configuration:\n")

def select_action(params, policy_net,  nn_state, mask, key):
    probs = policy_net.apply(params, nn_state, mask)
    return jax.random.choice(key, a=3, p=probs)
def get_closer(state, player):

    dists = []
    for action in range(env.num_actions):
        next_state, _, _, info = env.step_stack(state, action,player)
        #get distance between players
        dist = np.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2])
        dists.append(dist)
    a = np.argmin(dists)
    return a

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
        return masked_logits
    

policy_net = hk.without_apply_rng(hk.transform(policy_network))
key = jax.random.PRNGKey(1)
epsilon = 0.1

#Load data (deserialize)
#bounded
stack_folder = 'data/jax_drone/'
with open(stack_folder+'jax_stack_defender_wide.pickle', 'rb') as handle:
    params_defender = pickle.load(handle)
with open(stack_folder+'jax_stack_attacker_wide.pickle', 'rb') as handle:
    params_attacker = pickle.load(handle)

config = load_config("configs/config.yml")
print_config(config)

game_type = config['game']['type']
timestamp = str(datetime.datetime.now())





def get_true_state(attacker_cf, defender_cf):
    attacker_pos = attacker_cf.position()#
    attacker_x = attacker_pos[0]*2
    attacker_y = attacker_pos[1]*2
    attacker_yaw = float(attacker_cf.yaw()[0])

    defender_pos = defender_cf.position()#
    defender_x = defender_pos[0]*2
    defender_y = defender_pos[1]*2
    defender_yaw = float(defender_cf.yaw()[0])

    if np.linalg.norm(attacker_pos - defender_pos) < 0.4:
        print('unsafe!')
        attacker_cf.notifySetpointsStop()
        defender_cf.notifySetpointsStop()

        global timeHelper
        timeHelper.sleep(0.1)

        attacker_cf.land(0., 3)
        defender_cf.land(0., 3)
        raise Exception

    state = {'defender':np.array([defender_x, defender_y, defender_yaw]),'attacker':np.array([attacker_x, attacker_y, attacker_yaw])}

    return state

import math
def angle_interpolate_radians(angle1, angle2, lamb):
    diff = angle2 - angle1

    # Adjust for angles going beyond the standard range of 0 - 2*pi
    if diff > math.pi:
        angle2 -= 2 * math.pi 
    elif diff < -math.pi:
        angle2 += 2 * math.pi

    return angle1 + lamb * angle2
#
    timeHelper = swarm.timeHelper

key = jax.random.PRNGKey(1) #great seed




def main():




    global key
    global timeHelper
    states = []
    dones = []
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    defender_cf = swarm.allcfs.crazyflies[0]
    attacker_cf = swarm.allcfs.crazyflies[1]

    key = jax.random.PRNGKey(1) #great seed

    _, init_state, nn_state = env.reset(key)
    real = True


    if real:
        attacker_cf.setLEDColor(0, 0, 0)
        defender_cf.setLEDColor(0, 0, 0)

    print('init state', init_state)
    # timeHelper.sleep(0.5)
    # state = get_true_state(attacker_cf, defender_cf)
    # print(state)
    # exit()
    defender_cf.takeoff(1.0, 2.5)
    attacker_cf.takeoff(1.0, 2.5)
    

    timeHelper.sleep(3.5)

    initial_position_d = (init_state['defender'][0]/2, init_state['defender'][1]/2, 1.)
    initial_position_a = (init_state['attacker'][0]/2, init_state['attacker'][1]/2, 1.)

    defender_cf.goTo(initial_position_d, init_state['defender'][2], 3.5)
    attacker_cf.goTo(initial_position_a, init_state['attacker'][2], 3.5)
    timeHelper.sleep(5.)
    if real:
        attacker_cf.setLEDColor(0, 0, 255)
        defender_cf.setLEDColor(255, 0, 0)

    state = get_true_state(attacker_cf, defender_cf)
    state['attacker'][2] = init_state['attacker'][2]
    state['defender'][2] = init_state['defender'][2]
    print('real state', state)
    # exit()
    
        
    timesteps = np.arange(0, 60, 0.1)
    for t in timesteps:
        done_fail = False
        if t % 0.5 == 0:
            prev_state = copy.deepcopy(state) 
            for player in env.players:
                #nn_state = env.encode_helper(state)
                key, subkey = jax.random.split(key)
                if player == 'attacker':
                    mask = env.get_legal_actions_mask1(state)
                    if jnp.sum(mask) == 0:
                        print('FAIL')
                        done_fail = True
                    
                else:
                    mask = jnp.array([1,1,1])
                if sum(mask) != 0:
                    if player == 'defender': 
                        #action = jax.random.choice(subkey, 3)
                        action = select_action(params_defender, policy_net,  nn_state, mask, key)
                        #action = get_closer(state, player)
                        try:
                            true_state = get_true_state(attacker_cf, defender_cf)
                        except Exception:
                            done = True
                            break
                        true_state['attacker'][2] = state['attacker'][2]
                        true_state['defender'][2] = state['defender'][2]
                        print('true_state', true_state)
                        state, nn_state, reward, done_win = env.step_stack(true_state, action, player)
                    elif player=='attacker': 
                        action = select_action(params_attacker, policy_net,  nn_state, mask, key)
                        state, nn_state, reward, done_win = env.step_stack(state, action, player)
                        states.append(state)
                        #action = jax.random.choice(key, 3)
                    print(player, action)

                    

                    print('sim state', state)
                    done = jnp.logical_or(done_fail, done_win)

                    dones.append(done)
                    # print('prev_state', prev_state)
                    # print('state:', state)
                    # print('action:', action)
                else: 
                    done = True

        if done: 
            if real:
                attacker_cf.setLEDColor(255, 255, 0)
                defender_cf.setLEDColor(255, 255, 0)
                break
        
            

        attacker_prev_state = prev_state['attacker']
        defender_prev_state = prev_state['defender']

        attacker_state = state['attacker']
        defender_state = state['defender']

        lamb = (t % 0.5) * 2
        attacker_pos = lamb * attacker_state + (1 - lamb) * attacker_prev_state  
        defender_pos = lamb * defender_state + (1 - lamb) * defender_prev_state

        print("defender states", defender_prev_state, defender_state, defender_pos)

        

        # defender_cf.cmdPosition((defender_pos[0]/2, defender_pos[1]/2, 1.), float(defender_pos[2]))
        # attacker_cf.cmdPosition((attacker_pos[0]/2, attacker_pos[1]/2, 1.), float(attacker_pos[2]))
        defender_cf.cmdPosition((defender_pos[0]/2, defender_pos[1]/2, 1.), 0.)
        attacker_cf.cmdPosition((attacker_pos[0]/2, attacker_pos[1]/2, 1.), 0.)

        timeHelper.sleepForRate(20)
        # send position to crazyflie
        # wait for 0.1 seconds
    defender_cf.notifySetpointsStop()
    attacker_cf.notifySetpointsStop()
    timeHelper.sleep(0.05)

    pos = np.array(defender_cf.initialPosition) + np.array([0., 0., 1.0])
    defender_cf.land(0.04, 2.5)
    pos = np.array(attacker_cf.initialPosition) + np.array([0., 0., 1.0])
    attacker_cf.land(0.04,2.5)
    timeHelper.sleep(6.0)
    if real:
        attacker_cf.setLEDColor(0, 0, 0)
        defender_cf.setLEDColor(0, 0, 0)

    timeHelper.sleep(5.0)

    env.render(states, dones)
    env.make_gif('gifs/lab/jax_sim_real.gif')
    

if __name__ == '__main__':
    main()
    #ros2 launch crazyflie launch.py backend:=sim
    #adjust crazyfies.yaml
    #defender_cf.position() pose array x,y,z not yaw
    #ros2 run crazyflie_examples nice_hover 



