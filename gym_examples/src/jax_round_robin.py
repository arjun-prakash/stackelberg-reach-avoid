# Implement the REINFORCE algorithm
import sys
from collections import Counter
from pprint import pprint

from PIL import Image
from tqdm import tqdm

sys.path.append("..")
import datetime
import pickle

import haiku as hk
import imageio
import jax
import jax.numpy as jnp
import cvxpy as cp

import matplotlib
import matplotlib.pyplot as plt
# import gymnasium as gym
import numpy as np
import optax
from envs.two_player_dubins_car_ca import ContinuousTwoPlayerEnv
from matplotlib import cm
from PIL import Image
#from torch.utils.tensorboard import SummaryWriter
import yaml
from functools import partial
from jax_tqdm import scan_tqdm, loop_tqdm

from jax import config
from nashpy import Game
#config.update('jax_platform_name', 'gpu')
#config.update('jax_disable_jit', True)

#config.update('jax_enable_x64', True)
import pandas as pd

def breakpoint_if_contains_false(x):
  has_false = jnp.any(jnp.logical_not(x))
  def true_fn(x):
    jax.debug.breakpoint()

  def false_fn(x):
    pass
  jax.lax.cond(has_false, true_fn, false_fn, x)

STEPS_IN_EPISODE = 50


ENV = ContinuousTwoPlayerEnv(
        size=3,
        max_steps=50,
        capture_radius=0.3,
        goal_position=[0,-3],
        goal_radius=1,
        timestep=1,
        v_max=0.25,
    )     








def random_policy(key):
    """
    A simple random policy for selecting actions.
    :param key: JAX PRNG key.
    :param action_space: The action space of the environment.
    :param env_params: Additional environment parameters, if needed.
    :return: A randomly selected action.
    """
    # Assuming a discrete action space. Modify if the action space is different.
    num_actions = 3
    action = jax.random.randint(key, shape=(), minval=0, maxval=num_actions)
    return action



def select_action(params, policy_net,  nn_state, key):
    x,y = policy_net.apply(params, nn_state)
    return x,y

def get_closer(state, player):
    dists = []
    for action in range(ENV.num_actions):
        next_state, _, _, info = ENV.step_stack(state, action,player)
        #get distance between players
        dist = np.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2])
        dists.append(dist)
    a = np.argmin(dists)
    return a



def random_defender_policy(rng_key, min_magnitude=0.25):
    # Generate random values for x and y components of the action vector
    action_x = jax.random.uniform(rng_key, minval=-1.0, maxval=1.0)
    action_y = jax.random.uniform(rng_key, minval=-1.0, maxval=1.0)
    
    # Create the action vector
    action = jnp.array([action_x, action_y])
    
    # Calculate the magnitude of the action vector
    magnitude = jnp.linalg.norm(action)
    
    # Scale the action vector to have a magnitude greater than min_magnitude using jax.lax.cond
    scaled_action = jax.lax.cond(magnitude < min_magnitude,
                                 lambda _: action * (min_magnitude / magnitude),
                                 lambda _: action,
                                 None)
    
    return scaled_action


def get_closer(state):
    # Extract the positions of the defender and attacker from the state
    defender_pos = state['defender']  # Considering only x and y coordinates
    attacker_pos = state['attacker'] # Considering only x and y coordinates
    
    # Calculate the relative position vector from the defender to the attacker
    relative_pos = attacker_pos - defender_pos
    
    # Normalize the relative position vector to get the direction
    distance = jnp.linalg.norm(relative_pos)
    direction = relative_pos / (distance + 1e-8)
    
    # Set the defender's action as the normalized direction vector
    defender_action = direction
    
    return defender_action




#@partial(jax.jit, static_argnums=1)
def rollout(rng_input, init_state, init_nn_state, params_defender, params_attacker, steps_in_episode):
    """
    Perform a rollout in a JAX-compatible environment using lax.scan for JIT compatibility.
    :param rng_input: JAX PRNG key for random number generation.
    :param env: The environment object with reset and step functions.
    :param env_params: Parameters specific to the environment.
    :param steps_in_episode: Number of steps in the episode.
    :param action_space: The action space of the environment.
    :return: Observations, actions, rewards, next observations, and dones from the rollout.
    """
    # Reset the environment
    def check_done(prev_done, next_done):
        "delay done by one step to get bonus"
        result = jnp.logical_or(prev_done, jnp.logical_and(prev_done, next_done))
        return result


    def policy_step(state_input, _):
        """lax.scan compatible step transition in JAX environment."""
        state, nn_state, prev_done, rng = state_input
        rng, rng_step = jax.random.split(rng)
        action_defender = select_action(params_defender, policy_net, nn_state, rng_step)
        #action_defender = get_closer(state)
        #action_defender = random_defender_policy(rng_step)
        cur_state, nn_state, reward, cur_done, g = env.step(state, action_defender, 'defender')
        #jax.debug.print(f'mask: {attacker_mask}')
        #breakpoint_if_contains_false(attacker_mask)
        #check if there are no legal actions left, done is true
        action_attacker = select_action(params_attacker, policy_net, nn_state, rng_step)
        next_state, next_nn_state, reward, next_done, g = env.step(cur_state, action_attacker, 'attacker')
        
        next_done = jnp.logical_or(cur_done, next_done)
        done = jnp.logical_or(prev_done, next_done)

        is_terminal = check_done(prev_done, next_done)




        carry = (next_state,next_nn_state, done, rng_step)


        return carry, (action_defender, action_attacker, next_state, nn_state,reward, is_terminal, g) #only carrying the attacker reward
    
    def calculate_discounted_returns(rewards, discount_factor, mask):
        # Step 1: Reverse rewards and mask
        rewards_reversed = jnp.flip(rewards, axis=0)
        mask_reversed = jnp.flip(mask, axis=0)
        
        # Step 2: Apply mask to rewards
        masked_rewards_reversed = rewards_reversed * mask_reversed
        
        # Step 3: Calculate discounted rewards
        discounts = discount_factor ** jnp.arange(masked_rewards_reversed.size)
        discounted_rewards_reversed = masked_rewards_reversed * jnp.flip(discounts)
        #print(discounted_rewards_reversed)
        
        # Step 4: Cumulative sum for reversed rewards
        cumulative_rewards_reversed = jnp.cumsum(discounted_rewards_reversed)
        
        # Step 5: Reverse cumulative returns to original order
        cumulative_returns = jnp.flip(cumulative_rewards_reversed, axis=0)
        
        return cumulative_returns
    
   


    env = ENV

    policy_net = hk.without_apply_rng(hk.transform(policy_network))


    rng_reset, rng_episode = jax.random.split(rng_input)
    #jax.debug.print(f'rng: {rng_input}')
    #jax.debug.print(f'init_state: {init_nn_state}')
    
    
    #if not set_env reset
    #obs, state, nn_state = env.reset(rng_input) #add a cond here for setting the state

    


    # Scan over the episode step loop
    initial_carry = (init_state, init_nn_state, False, rng_episode) #rng_episode
    
    _, scan_out = jax.lax.scan(policy_step, initial_carry, None, length=steps_in_episode)

    # Unpack scan output
    actions_defender,actions_attacker, states, nn_states, rewards, dones, gs = scan_out
    mask = jnp.logical_not(dones)
    #update networks


    returns = calculate_discounted_returns(rewards, 0.99, mask)


    return actions_defender, actions_attacker, states, nn_states, returns, dones, rewards, gs














def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def print_config(config):
    print("Starting experiment with the following configuration:\n")
    print(yaml.dump(config))


def convert_state_to_list(state_dict):
    """
    Convert a dictionary of states to a list of state dictionaries.

    :param state_dict: A dictionary where keys are player identifiers and values are arrays of states.
    :return: A list where each element is a state dictionary corresponding to a time step.
    """
    num_steps = state_dict['attacker'].shape[0]  # Assuming both players have the same number of steps
    state_list = []

    for i in range(num_steps):
        step_state = {player: state_dict[player][i] for player in state_dict}
        state_list.append(step_state)

    return state_list

def batched_env_reset(rng_keys):
        # Vectorize the existing env.reset function to handle batched PRNG keys
        batched_reset = jax.vmap(env.reset)

        
        # Call the vectorized reset function with the batched PRNG keys
        init_obs, initial_states, nn_states = batched_reset(rng_keys)
        
        return init_obs, initial_states, nn_states
    


def policy_network_nash(observation):
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
            hk.Linear(ENV.num_actions),
            jax.nn.softmax,
        ]
    )
    return net(observation)

def policy_network(observation):
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
            hk.Linear(2),
        ]
    )
    return net(observation)






import jax
import jax.numpy as jnp
import itertools


import jax
import jax.numpy as jnp
import itertools





# Example usage:
config = load_config("configs/config.yml")
    
game_type = config['game']['type']
timestamp = str(datetime.datetime.now())
folder = 'data/jax_round_robin/'
# #make folder
# import os
# if not os.path.exists(folder):
#     os.makedirs(folder)





 

rng_input = jax.random.PRNGKey(1)
#actions_defender, actions_attacker, states,rewards, dones = rollout(rng_input, steps_in_episode)

env = ENV

obs, state ,nn_state = env.reset(rng_input)
state = env.encode_helper(state)
print(state.shape)




    
        
# stack_folder = 'data/jax_stack/2024-02-20 16:45:45.540627/'
# nash_folder = 'data/jax_nash/2024-02-20 16:59:48.570596/'
# pe_folder = 'data/jax_pe_stack/2024-02-20 17:35:27.995539/'

#bounded
stack_folder = 'data/jax_cont_stack2/2024-03-12 18:33:43.286607/'
nash_folder = 'data/jax_cont_nash/2024-03-12 18:44:31.030119/'


with open(stack_folder+'jax_stack_defender.pickle', 'rb') as handle:
    params_defender = pickle.load(handle)
# with open(stack_folder+'jax_stack_attacker.pickle', 'rb') as handle:
#     params_attacker = pickle.load(handle)

# with open(nash_folder+'jax_stack_defender.pickle', 'rb') as handle:
#     params_defender = pickle.load(handle)
with open(nash_folder+'jax_stack_attacker.pickle', 'rb') as handle:
    params_attacker = pickle.load(handle)

# with open(pe_folder+'jax_stack_defender.pickle', 'rb') as handle:
#     params_defender = pickle.load(handle)

#rng_input = jax.random.PRNGKey(69679898967)
rng_input = jax.random.PRNGKey(0)
policy_net = hk.without_apply_rng(hk.transform(policy_network))
#params_defender = policy_net.init(rng_input, nn_state)
# params_attacker = policy_net.init(rng_input, nn_state, jnp.array([1,1,1]))






   


    




#do a rollout

obs, state, init_nn_state = env.reset(rng_input)


actions_defender, actions_attacker, states, nn_states, returns, dones, rewards, gs = rollout(rng_input, state, init_nn_state, params_defender, params_attacker,STEPS_IN_EPISODE)
mask = jnp.logical_not(dones)
#set mask to all 1



batched_rollout = jax.vmap(rollout, in_axes=(0,0,0, None, None, None), out_axes=0)
keys = jax.random.split(rng_input, num=100)

init_obs, initial_states, initial_nn_states = batched_env_reset(keys)
all_actions_defender, all_actions_attacker, all_states, all_nn_states, all_returns, all_dones, all_rewards, all_gs = batched_rollout(keys,  initial_states, initial_nn_states, params_defender, params_attacker, STEPS_IN_EPISODE)
all_masks = jnp.logical_not(all_dones)

#multiply by mask
all_rewards = all_rewards * all_masks
#check if any are positive
wins = [jnp.any(g < 0) for g in all_gs]
wins = jnp.array([item.item() for item in wins])
print('num wins',sum(wins))

win_masks = all_masks[wins]
loss_masks = all_masks[~wins]

print('average win length:', jnp.mean(jnp.sum(win_masks, axis=1)))
print('std win length:', jnp.std(jnp.sum(win_masks, axis=1)))
print('average loss length:', jnp.mean(jnp.sum(loss_masks, axis=1)))
print('std loss length:', jnp.std(jnp.sum(loss_masks, axis=1))) 



print('rendering')
states = convert_state_to_list(states)
env.render(states, dones)
env.make_gif(folder+'/rollout.gif')