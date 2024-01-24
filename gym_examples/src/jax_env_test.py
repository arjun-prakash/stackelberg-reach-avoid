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

import matplotlib
import matplotlib.pyplot as plt
# import gymnasium as gym
import numpy as np
import optax
from envs.two_player_dubins_car_jax import TwoPlayerDubinsCarEnv
from matplotlib import cm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import yaml
from functools import partial




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





#@partial(jax.jit, static_argnums=1)
def rollout(rng_input, params_defender, params_attacker, steps_in_episode):
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

    env = TwoPlayerDubinsCarEnv(
        game_type=game_type,
        num_actions=3,
        size=3,
        reward='',
        max_steps=5,
        init_defender_position=[0,0,0],
        init_attacker_position=[2,2,0],
        capture_radius=0.3,
        goal_position=[6,0],
        goal_radius=1,
        timestep=1,
        v_max=0.25,
        omega_max=30,
    )   


    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state, nn_state = env.reset(rng_reset)
    print('state',state)

    def policy_step(state_input, _):
        """lax.scan compatible step transition in JAX environment."""
        state, nn_state, prev_done, rng = state_input
        rng, rng_step = jax.random.split(rng)
        action_defender = random_policy(rng_step)
        next_state, nn_state, reward, next_done = env.step(rng_step, state, action_defender, 'defender')
        action_attacker = random_policy(rng_step)
        next_state, nn_state, reward, next_done = env.step(rng_step, next_state, action_attacker, 'attacker')
        done = jnp.logical_or(prev_done, next_done)
        carry = (next_state,nn_state, done, rng)


        return carry, (action_defender, action_attacker, next_state, nn_state,reward, done)


    # Scan over the episode step loop
    initial_carry = (state, nn_state, False, rng_episode)
    _, scan_out = jax.lax.scan(policy_step, initial_carry, None, length=steps_in_episode)

    # Unpack scan output
    actions_defender,actions_attacker, states, nn_states, rewards, dones = scan_out

    #update networks

    return actions_defender, actions_attacker, states, nn_states, rewards, dones

def rollout_body(i, carry):
    rng_input, params_defender, params_attacker, all_actions_defender, all_actions_attacker ,all_nn_states, all_rewards, all_dones = carry
    rng_input, subkey = jax.random.split(rng_input)
    actions_defender, actions_attacker, states, nn_states, rewards, dones = rollout(subkey, params_defender, params_attacker, 50)

    # Store the results of this rollout
    all_actions_defender = all_actions_defender.at[i, :].set(actions_defender)
    all_actions_attacker = all_actions_attacker.at[i, :].set(actions_attacker)
    all_nn_states = all_nn_states.at[i, :].set(nn_states) 
    all_rewards = all_rewards.at[i, :].set(rewards)
    all_dones = all_dones.at[i, :].set(dones)

    return rng_input, params_defender, params_attacker, all_actions_defender, all_actions_attacker, all_nn_states, all_rewards, all_dones











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
            hk.Linear(env.num_actions),
            jax.nn.softmax,
        ]
    )
    return net(observation)

@jax.jit
def train():

   
    
    env = TwoPlayerDubinsCarEnv(
            game_type=game_type,
            num_actions=3,
            size=3,
            reward='',
            max_steps=5,
            init_defender_position=[0,0,0],
            init_attacker_position=[2,2,0],
            capture_radius=0.3,
            goal_position=[6,0],
            goal_radius=1,
            timestep=1,
            v_max=0.25,
            omega_max=30,
        )   


    rng_input = jax.random.PRNGKey(0)
    #actions_defender, actions_attacker, states,rewards, dones = rollout(rng_input, steps_in_episode)
    steps_in_episode = 50
    num_episodes = 1000


    steps_in_episode = 50

    obs, state ,nn_state = env.reset(rng_input)
    state = env.encode_helper(state)
    print(state.shape)



    # Adjust the shape based on the actual structure of your state and action
    all_actions_defender = jnp.zeros((num_episodes, steps_in_episode), dtype=jnp.int32)
    all_actions_attacker = jnp.zeros((num_episodes, steps_in_episode), dtype=jnp.int32)
    all_states = jnp.zeros((num_episodes, steps_in_episode,3))  # Add the shape of your state
    all_nn_states = jnp.zeros((num_episodes, steps_in_episode,13))  # Add the shape of your state
    all_rewards = jnp.zeros((num_episodes, steps_in_episode))
    all_dones = jnp.zeros((num_episodes, steps_in_episode), dtype=jnp.bool_)


    ############################
        # Initialize Haiku policy network
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    params_defender = policy_net.init(rng_input, state)
    params_attacker = policy_net.init(rng_input, state)

    # Initialize Haiku optimizer
    optimizer_defender = optax.adam(1e-3)
    optimizer_attacker = optax.adam(1e-3)

    final_rng_input, params_defender, params_attacker,all_actions_defender, all_actions_attacker, all_nn_states, all_rewards, all_dones = \
            jax.lax.fori_loop(
                0, num_episodes, 
                rollout_body, 
                (rng_input, params_defender, params_attacker, all_actions_defender, all_actions_attacker,all_nn_states, all_rewards, all_dones)
            )
    
    return all_nn_states





# Example usage:
config = load_config("configs/config.yml")
    
game_type = config['game']['type']
timestamp = str(datetime.datetime.now())

env = TwoPlayerDubinsCarEnv(
        game_type=game_type,
        num_actions=3,
        size=3,
        reward='',
        max_steps=5,
        init_defender_position=[0,0,0],
        init_attacker_position=[2,2,0],
        capture_radius=0.3,
        goal_position=[6,0],
        goal_radius=1,
        timestep=1,
        v_max=0.25,
        omega_max=30,
    )   



rng_input = jax.random.PRNGKey(1)
steps_in_episode = 50
#actions_defender, actions_attacker, states,rewards, dones = rollout(rng_input, steps_in_episode)
steps_in_episode = 50
num_episodes = 1000


steps_in_episode = 50

obs, state ,nn_state = env.reset(rng_input)
state = env.encode_helper(state)
print(state.shape)


all_nn_states = train()
decoded = env.decode_helper(all_nn_states[0])





print('rendering')
# print(states)
# states = convert_state_to_list(states)
env.render(decoded)
env.make_gif('gifs/jax/test3.gif')

