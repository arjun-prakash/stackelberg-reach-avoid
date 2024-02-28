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
from envs.two_player_dubins_car_jax import TwoPlayerDubinsCarEnv
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


ENV = TwoPlayerDubinsCarEnv(
        game_type='stackelberg',
        num_actions=3,
        size=3,
        reward='',
        max_steps=5,
        init_defender_position=[0,0,0],
        init_attacker_position=[2,2,0],
        capture_radius=0.25,
        goal_position=[0,-3],
        goal_radius=1,
        timestep=1,
        v_max=0.25,
        omega_max=30,
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


def select_action_nash(params, policy_net,  nn_state, key):
    probs = policy_net_nash.apply(params, nn_state)
    return jax.random.choice(key, a=3, p=probs)

def select_action_stack(params, policy_net,  nn_state, mask, key):
    probs = policy_net_stack.apply(params, nn_state, mask)
    return jax.random.choice(key, a=3, p=probs)

def get_closer(state, player):
    dists = []
    for action in range(ENV.num_actions):
        next_state, _, _, info = ENV.step_stack(state, action,player)
        #get distance between players
        dist = np.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2])
        dists.append(dist)
    a = np.argmin(dists)
    return a

def get_closer2(state, player):
            # Define a helper function to check if the 'attacker' gets captured by the 'defender'
        def apply_actions(state, action):
            next_state, nn_state, reward, done = ENV.step_stack(state, action, 'defender')
            #check reward is -200 and done is true
            return jnp.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2])
        
        # JIT compile the helper function for efficiency

        attacker_actions = jnp.array([0,1,2])

        # Vectorize the apply_actions function to handle all action pairs in parallel
        batch_apply_actions = jax.vmap(apply_actions, in_axes=(None, 0))

        # Apply all action pairs to the initial state

        dists = batch_apply_actions(state, attacker_actions)
        a = jnp.argmin(dists)
        
        
        return a




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


    def policy_step_stack_v_stack(state_input, _):
        """lax.scan compatible step transition in JAX environment."""
        state, nn_state, prev_done, rng = state_input
        rng, rng_step = jax.random.split(rng)
        defender_mask = jnp.array([1,1,1])
        action_defender = select_action_stack(params_defender, policy_net_stack, nn_state, defender_mask, rng_step)
        action_defender =  0 #get_closer2(state, 'defender')
        cur_state, cur_nn_state, reward, cur_done = env.step_stack(state, action_defender, 'defender')
        attacker_mask = ENV.get_legal_actions_mask1(cur_state)
        #print(attacker_mask)

        no_moves_done = jax.lax.cond(jnp.all(attacker_mask == 0), lambda x: True, lambda x: False, None)
        action_attacker = select_action_stack(params_attacker, policy_net_stack, cur_nn_state, attacker_mask, rng_step)
        next_state, next_nn_state, reward, next_done = env.step_stack(cur_state, action_attacker, 'attacker')
        
        next_done = jnp.logical_or(no_moves_done, next_done)
        next_done = jnp.logical_or(False, next_done)
        done = jnp.logical_or(prev_done, next_done)

        is_terminal = check_done(prev_done, next_done)
        carry = (next_state,next_nn_state, done, rng_step)


        return carry, (action_defender, action_attacker, next_state, nn_state,reward, is_terminal, attacker_mask) #only carrying the attacker reward
    

    def policy_step_stack_v_nash(state_input, _):
        """lax.scan compatible step transition in JAX environment."""
        state, nn_state, prev_done, rng = state_input
        rng, rng_step = jax.random.split(rng)
        action_defender = select_action_nash(params_defender, policy_net_stack, nn_state, rng_step)
        cur_state, cur_nn_state, reward, cur_done = env.step_stack(state, action_defender, 'defender')
        attacker_mask = ENV.get_legal_actions_mask1(cur_state)

        no_moves_done = jax.lax.cond(jnp.all(attacker_mask == 0), lambda x: True, lambda x: False, None)
        action_attacker = select_action_stack(params_attacker, policy_net_nash, cur_nn_state, attacker_mask, rng_step)
        next_state, next_nn_state, reward, next_done = env.step_stack(cur_state, action_attacker, 'attacker')
        
        next_done = jnp.logical_or(no_moves_done, next_done)
        next_done = jnp.logical_or(False, next_done)
        done = jnp.logical_or(prev_done, next_done)

        is_terminal = check_done(prev_done, next_done)
        carry = (next_state,next_nn_state, done, rng_step)


        return carry, (action_defender, action_attacker, next_state, nn_state,reward, is_terminal, attacker_mask) #only carrying the attacker reward
    
    def policy_step_nash_v_stack(state_input, _):
        """lax.scan compatible step transition in JAX environment."""
        state, nn_state, prev_done, rng = state_input
        rng, rng_step = jax.random.split(rng)
        defender_mask = jnp.array([1,1,1])
        action_defender = select_action_stack(params_defender, policy_net_stack, nn_state, defender_mask, rng_step)
        cur_state, cur_nn_state, reward, cur_done = env.step_nash(state, action_defender, 'defender')
        attacker_mask = ENV.get_legal_actions_mask1(cur_state)
        #no_moves_done = jax.lax.cond(jnp.all(attacker_mask == 0), lambda x: True, lambda x: False, None)

        action_attacker = select_action_nash(params_attacker, policy_net_nash, cur_nn_state, rng_step)
        next_state, next_nn_state, reward, next_done = env.step_nash(cur_state, action_attacker, 'attacker')
        
        next_done = jnp.logical_or(cur_done, next_done)
        done = jnp.logical_or(prev_done, next_done)

        is_terminal = check_done(prev_done, next_done)
        carry = (next_state,next_nn_state, done, rng_step)


        return carry, (action_defender, action_attacker, next_state, nn_state,reward, is_terminal, attacker_mask) #only carrying the attacker reward

    def policy_step_nash_v_nash(state_input, _):
        """lax.scan compatible step transition in JAX environment."""
        state, nn_state, prev_done, rng = state_input
        rng, rng_step = jax.random.split(rng)
        #rng_step = rng
        #action_defender = random_policy(rng_step)
        action_defender = select_action_nash(params_defender, policy_net_nash, nn_state, rng_step)
        action_defender =  get_closer2(state, 'defender')
        next_state, _, reward, cur_done = env.step_nash(state, action_defender, 'defender')
        #action_attacker = random_policy(rng_step)
        action_attacker =select_action_nash(params_attacker, policy_net_nash, nn_state, rng_step)
        next_state, next_nn_state, reward, next_done = env.step_nash(next_state, action_attacker, 'attacker')
        next_done = jnp.logical_or(cur_done, next_done)
        done = jnp.logical_or(prev_done, next_done)
        is_terminal = check_done(prev_done, next_done)




        carry = (next_state,next_nn_state, done, rng_step)


        return carry, (action_defender, action_attacker, next_state, nn_state,reward, is_terminal, None) #only carrying the attacker reward
    
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

    policy_net_stack = hk.without_apply_rng(hk.transform(policy_network_stack))
    #value_net = hk.without_apply_rng(hk.transform(value_network))


    rng_reset, rng_episode = jax.random.split(rng_input)
    #jax.debug.print(f'rng: {rng_input}')
    #jax.debug.print(f'init_state: {init_nn_state}')
    
    
    #if not set_env reset
    #obs, state, nn_state = env.reset(rng_input) #add a cond here for setting the state

    


    # Scan over the episode step loop
    initial_carry = (init_state, init_nn_state, False, rng_episode) #rng_episode
    
    _, scan_out = jax.lax.scan(policy_step_nash_v_nash, initial_carry, None, length=steps_in_episode)

    # Unpack scan output
    actions_defender,actions_attacker, states, nn_states, rewards, dones, attacker_masks = scan_out
    mask = jnp.logical_not(dones)
    #update networks


    returns = calculate_discounted_returns(rewards, 0.99, mask)


    return actions_defender, actions_attacker, states, nn_states, returns, dones, rewards, attacker_masks











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

def policy_network_stack(observation, legal_moves):
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

        logits = net(observation)
        masked_logits = jnp.where(legal_moves, logits, 1e-8)
        return masked_logits






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
stack_folder = 'data/jax_stack/2024-02-26 17:40:36.168512/'
nash_folder = 'data/jax_nash/2024-02-26 16:32:19.316452/'


# with open(stack_folder+'jax_stack_defender.pickle', 'rb') as handle:
#     params_defender = pickle.load(handle)
# with open(stack_folder+'jax_stack_attacker.pickle', 'rb') as handle:
#     params_attacker = pickle.load(handle)

with open(nash_folder+'jax_nash_defender.pickle', 'rb') as handle:
    params_defender = pickle.load(handle)
with open(nash_folder+'jax_nash_attacker.pickle', 'rb') as handle:
    params_attacker = pickle.load(handle)

# with open(pe_folder+'jax_stack_defender.pickle', 'rb') as handle:
#     params_defender = pickle.load(handle)

#rng_input = jax.random.PRNGKey(69679898967)
rng_input = jax.random.PRNGKey(0)
policy_net_stack = hk.without_apply_rng(hk.transform(policy_network_stack))
policy_net_nash = hk.without_apply_rng(hk.transform(policy_network_nash))
# params_defender = policy_net.init(rng_input, nn_state, jnp.array([1,1,1]))
# params_attacker = policy_net.init(rng_input, nn_state, jnp.array([1,1,1]))






   


    




#do a rollout

obs, state, init_nn_state = env.reset(rng_input)


actions_defender, actions_attacker, states, nn_states, returns, dones, rewards, masks = rollout(rng_input, state, init_nn_state, params_defender, params_attacker,STEPS_IN_EPISODE)
mask = jnp.logical_not(dones)
#set mask to all 1



batched_rollout = jax.vmap(rollout, in_axes=(0,0,0, None, None, None), out_axes=0)
keys = jax.random.split(rng_input, num=100)

init_obs, initial_states, initial_nn_states = batched_env_reset(keys)
all_actions_defender, all_actions_attacker, all_states, all_nn_states, all_returns, all_dones, all_rewards, all_attacker_masks = batched_rollout(keys,  initial_states, initial_nn_states, params_defender, params_attacker, STEPS_IN_EPISODE)
all_masks = jnp.logical_not(all_dones)

#multiply by mask
all_rewards = all_rewards * all_masks
#check if any are positive
wins = [jnp.any(x > 0) for x in all_rewards]
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