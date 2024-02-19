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
import pandas as pd

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

# config.update('jax_enable_x64', True)

STEPS_IN_EPISODE = 50
BATCH_SIZE = 32
NUM_ITERS = 6000
is_train = True


ENV = TwoPlayerDubinsCarEnv(
        game_type='nash',
        num_actions=3,
        size=3,
        reward='',
        max_steps=50,
        init_defender_position=[0,0,0],
        init_attacker_position=[2,2,0],
        capture_radius=0.3,
        goal_position=[0,-3],
        goal_radius=1,
        timestep=1,
        v_max=0.25,
        omega_max=30,
    )   



def get_q_matrix(rng_input, state, nn_state, params_defender, params_attacker, steps_in_episode):

    env = ENV

    def apply_actions(state, actions, rng_input):
        action_defender, action_attacker = actions
        next_state, nn_state, _, _ = env.step_nash(state, action_defender, 'defender')
        init_state, init_nn_state, reward, _ = env.step_nash(next_state, action_attacker, 'attacker')
        return init_state, init_nn_state, reward
    


    defender_actions = jnp.array([0,1,2])
    attacker_actions = jnp.array([0,1,2])
    action_pairs = jnp.array(list(itertools.product(defender_actions, attacker_actions)))

    # Vectorize the apply_actions function to handle all action pairs in parallel
    batch_apply_actions = jax.vmap(apply_actions, in_axes=(None, 0, None))

    # Apply all action pairs to the initial state
    init_states, init_nn_states, rewards = batch_apply_actions(state, action_pairs, rng_input)
    rewards = jnp.array(rewards).reshape(len(defender_actions), len(attacker_actions))
    #print('init_nn_states q', init_nn_states)
    # Ensure your rollout function can handle batched inputs
    # Vectorize the rollout function to handle all initial states from action pairs in parallel
    batched_rollout = jax.vmap(rollout, in_axes=(None,0,0, None, None, None))

    # Perform batched rollouts
    _, _, states, _, returns, _, rewards_traj = batched_rollout(rng_input, init_states, init_nn_states, params_defender, params_attacker,STEPS_IN_EPISODE-1)


    #get the first return for each action pair
    returns = jnp.array([r[0] for r in returns])


    # Reshape Q-values into a matrix
    #row player is defener, 
    q_matrix = returns.reshape(len(defender_actions), len(attacker_actions))
    q_matrix = rewards +  .99*q_matrix


    return q_matrix


def get_q_matrix_value_net(rng_input, state, nn_state, params_defender, params_attacker, params_value, steps_in_episode):

    env = ENV

    def apply_actions(state, actions, rng_input):
        action_defender, action_attacker = actions
        next_state, nn_state, _, _ = env.step_nash(state, action_defender, 'defender')
        init_state, init_nn_state, reward, _ = env.step_nash(next_state, action_attacker, 'attacker')
        return init_state, init_nn_state, reward
    


    defender_actions = jnp.array([0,1,2])
    attacker_actions = jnp.array([0,1,2])
    action_pairs = jnp.array(list(itertools.product(defender_actions, attacker_actions)))

    # Vectorize the apply_actions function to handle all action pairs in parallel
    batch_apply_actions = jax.vmap(apply_actions, in_axes=(None, 0, None))

    # Apply all action pairs to the initial state
    init_states, init_nn_states, rewards = batch_apply_actions(state, action_pairs, rng_input)
    rewards = jnp.array(rewards).reshape(len(defender_actions), len(attacker_actions))
    #print('init_nn_states q', init_nn_states)
    # Ensure your rollout function can handle batched inputs
    # Vectorize the rollout function to handle all initial states from action pairs in parallel
    batched_value_estimate = jax.vmap(value_network.apply, in_axes=(None, 0))

    # Perform batched rollouts
    _, _, states, _, returns, _, rewards_traj = batched_rollout(rng_input, init_states, init_nn_states, params_defender, params_attacker,STEPS_IN_EPISODE-1)


    #get the first return for each action pair
    returns = jnp.array([r[0] for r in returns])


    # Reshape Q-values into a matrix
    #row player is defener, 
    q_matrix = returns.reshape(len(defender_actions), len(attacker_actions))
    q_matrix = rewards +  .99*q_matrix


    return q_matrix

def solve_nash(q_matrix, v):

    g = Game(q_matrix.T)
    eqs = list(g.support_enumeration())
    if len(eqs) > 0:
        payoff =  jnp.array(g[eqs[0]][0])
    else:
        payoff =  jnp.nan

    error = payoff - v
    #print('payoff', payoff)

    return abs(error[0])


@jax.jit
def get_q_and_v(rng_input, params_defender, params_attacker):


 

    obs, state, nn_state = ENV.reset(rng_input)
    batched_get_q_matrix = jax.vmap(get_q_matrix, in_axes=(0, None, None, None, None, None))
    batched_rollout = jax.vmap(rollout, in_axes=(0,None,None, None, None, None), out_axes=0)
    keys = jax.random.split(rng_input, num=BATCH_SIZE*2)
    q_matricies = batched_get_q_matrix(keys, state, nn_state, params_defender, params_attacker, STEPS_IN_EPISODE)
    rollouts = batched_rollout(keys, state, nn_state, params_defender, params_attacker, STEPS_IN_EPISODE)

    q_matrix = jnp.mean(q_matricies, axis=0)
    returns = jnp.mean(jnp.array([r[0] for r in rollouts[4]]))

    return q_matrix, returns





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
    probs = policy_net.apply(params, nn_state)
    probs = probs
    return jax.random.choice(key, a=3, p=probs)






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
        #rng_step = rng
        #action_defender = random_policy(rng_step)
        action_defender = select_action(params_defender, policy_net, nn_state, rng_step)
        next_state, _, reward, cur_done = env.step_nash(state, action_defender, 'defender')
        #action_attacker = random_policy(rng_step)
        action_attacker =select_action(params_attacker, policy_net, nn_state, rng_step)
        next_state, next_nn_state, reward, next_done = env.step_nash(next_state, action_attacker, 'attacker')
        next_done = jnp.logical_or(cur_done, next_done)
        done = jnp.logical_or(prev_done, next_done)
        is_terminal = check_done(prev_done, next_done)




        carry = (next_state,next_nn_state, done, rng_step)


        return carry, (action_defender, action_attacker, next_state, nn_state,reward, is_terminal) #only carrying the attacker reward
    
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
    value_net = hk.without_apply_rng(hk.transform(value_network))


    rng_reset, rng_episode = jax.random.split(rng_input)
    #jax.debug.print(f'rng: {rng_input}')
    #jax.debug.print(f'init_state: {init_nn_state}')
    
    
    #if not set_env reset
    #obs, state, nn_state = env.reset(rng_input) #add a cond here for setting the state

    


    # Scan over the episode step loop
    initial_carry = (init_state, init_nn_state, False, rng_episode) #rng_episode
    
    _, scan_out = jax.lax.scan(policy_step, initial_carry, None, length=steps_in_episode)

    # Unpack scan output
    actions_defender,actions_attacker, states, nn_states, rewards, dones = scan_out
    mask = jnp.logical_not(dones)
    #update networks


    returns = calculate_discounted_returns(rewards, 0.99, mask)


    return actions_defender, actions_attacker, states, nn_states, returns, dones, rewards








def save_params(defender_params, attacker_params, value_params, folder):
    with open(folder+'/jax_nash_defender.pickle', 'wb') as handle:
        pickle.dump(defender_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+'/jax_nash_attacker.pickle', 'wb') as handle:
        pickle.dump(attacker_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+'/jax_nash_value.pickle', 'wb') as handle:
        pickle.dump(value_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('params saved')


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
            hk.Linear(ENV.num_actions),
            jax.nn.softmax,
        ]
    )
    return net(observation)

def value_network(observation):
        net = hk.Sequential(
            [
                hk.Linear(64),
                jax.nn.relu,
                hk.Linear(64),
                jax.nn.relu,
                hk.Linear(1),
            ]
        )
        return net(observation)





@jax.jit
def train():


    @jax.jit
    def loss_attacker(
        params, value_params, observations, actions, returns, padding_mask
    ):
        action_probabilities = policy_net.apply(params, observations)
        #action_probabilities = jax.nn.softmax(action_probabilities)
        log_probs = jnp.log(jnp.take_along_axis(
                action_probabilities + 10e-6, actions[..., None], axis=-1
            )
        )

        # Get baseline values
        baseline_values = value_net.apply(value_params, observations).squeeze(-1)
        advantage = returns - baseline_values
            
        log_probs = log_probs.reshape(returns.shape)
        masked_loss = padding_mask * (-log_probs * jax.lax.stop_gradient(advantage))
        return jnp.sum(masked_loss) / jnp.sum(padding_mask)

    @jax.jit
    def loss_defender(
        params, value_params, observations, actions, returns, padding_mask
    ):
        action_probabilities = policy_net.apply(params, observations)
        #action_probabilities = jax.nn.softmax(action_probabilities)
        log_probs = jnp.log(jnp.take_along_axis(
                action_probabilities + 10e-6, actions[..., None], axis=-1
            )
        )
        # Get baseline values
        baseline_values = value_net.apply(value_params, observations).squeeze(-1)
        advantage = returns - baseline_values

        log_probs = log_probs.reshape(returns.shape)
        masked_loss = padding_mask * (log_probs * jax.lax.stop_gradient(advantage))
        return jnp.sum(masked_loss) / jnp.sum(padding_mask)

# Define update function

    @jax.jit
    def update_defender(
        params, value_params, opt_state, observations, actions, returns, padding_mask
    ):
        grads = jax.grad(loss_defender)(
            params, value_params, observations, actions, returns, padding_mask
        )
        updates, opt_state = optimizer_defender.update(
            grads, params=params, state=opt_state
        )
        return optax.apply_updates(params, updates), opt_state, grads

    @jax.jit
    def update_attacker(
        params, value_params, opt_state, observations, actions, returns, padding_mask
    ):
        grads = jax.grad(loss_attacker)(
            params, value_params, observations, actions, returns, padding_mask
        )
        updates, opt_state = optimizer_attacker.update(
            grads, params=params, state=opt_state
        )
        return optax.apply_updates(params, updates), opt_state, grads

    @jax.jit
    def value_loss(params_value, observations, returns, padding_mask):
        predicted_values = value_net.apply(params_value, observations).squeeze(-1)
        #predicted_values
        # Calculate MSE loss
        loss = jnp.mean(padding_mask * (predicted_values - returns) ** 2)
        return loss

    @jax.jit
    def update_value_network(params_value, opt_state_value, observations, returns, padding_mask):
        grads_value = jax.grad(value_loss)(params_value, observations, returns, padding_mask)
        updates_value, opt_state_value = value_optimizer.update(grads_value, opt_state_value)
        new_params_value = optax.apply_updates(params_value, updates_value)
        return new_params_value, opt_state_value
    
    def batched_env_reset(rng_keys):
        # Vectorize the existing env.reset function to handle batched PRNG keys
        batched_reset = jax.vmap(env.reset)

        
        # Call the vectorized reset function with the batched PRNG keys
        init_obs, initial_states, nn_states = batched_reset(rng_keys)
        
        return init_obs, initial_states, nn_states

    
    @loop_tqdm(NUM_ITERS)
    def training_step(i, train_state):


        # Unpack the training state
        params_defender, params_attacker, params_value, opt_state_defender, opt_state_attacker, opt_state_value, rng_input, metrics, q_values, v_values  = train_state
        rng_input, subkey = jax.random.split(rng_input)

      


        batched_rollout = jax.vmap(rollout, in_axes=(0,0,0, None, None, None), out_axes=0)

        keys = jax.random.split(subkey, num=BATCH_SIZE)
        init_obs, initial_states, initial_nn_states = batched_env_reset(keys)

        all_actions_defender, all_actions_attacker, all_states, all_nn_states, all_returns, all_dones, all_rewards = batched_rollout(keys,  initial_states, initial_nn_states, params_defender, params_attacker, STEPS_IN_EPISODE)

        all_masks = jnp.logical_not(all_dones)


        # Update the attacker network
        params_attacker, opt_state_attacker, attacker_grads = update_attacker(
            params_attacker, params_value, opt_state_attacker, all_nn_states, all_actions_attacker, all_returns, all_masks
        )

        # Update the defender network
        params_defender, opt_state_defender, defender_grads = update_defender(
            params_defender, params_value, opt_state_defender, all_nn_states, all_actions_defender, all_returns, all_masks
        )
        
        # Update the value network
        params_value, opt_state_value = update_value_network(
            params_value, opt_state_value, all_nn_states, all_returns, all_masks
        )

        #get bellman error
        q, v = get_q_and_v(rng_input, params_defender, params_attacker)


        #get metrics
        # Calculate the average return across all episodes
        average_return = jnp.mean(jnp.sum(all_returns, axis=1))
        attacker_norm = optax.global_norm(attacker_grads)
        defender_norm = optax.global_norm(defender_grads)
        current_value_loss = value_loss(params_value, all_nn_states, all_returns, all_masks)
        #bellman_error = jax.lax.stop_gradient(calc_nash_bellman_error(rng_input, params_defender, params_attacker))

        m = (average_return, attacker_norm, defender_norm, current_value_loss)
        metrics = metrics.at[i, :].set(m)
        q_values = q_values.at[i, :].set(q)
        v_values = v_values.at[i, :].set(v)




        # Return updated training state
        return params_defender, params_attacker, params_value, opt_state_defender, opt_state_attacker, opt_state_value, subkey, metrics, q_values, v_values

    


    
    env = ENV

    rng_input = jax.random.PRNGKey(0)
    #actions_defender, actions_attacker, states,rewards, dones = rollout(rng_input, steps_in_episode)


    obs, state ,nn_state = env.reset(rng_input)
    state = env.encode_helper(state)


    


    ############################
        # Initialize Haiku policy network
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    params_defender = policy_net.init(rng_input, nn_state)
    params_attacker = policy_net.init(rng_input, nn_state)

    value_net = hk.without_apply_rng(hk.transform(value_network))
    params_value = value_net.init(rng_input, nn_state)

    # Initialize Haiku optimizer
    optimizer_defender = optax.radam(1e-4, b1=0.9, b2=0.9)
    optimizer_attacker = optax.radam(1e-4, b1=0.9, b2=0.9)
    value_optimizer = optax.radam(1e-4, b1=0.9, b2=0.9)

    # Initialize optimizer state
    opt_state_attacker = optimizer_attacker.init(params_attacker)
    opt_state_defender = optimizer_defender.init(params_defender)
    opt_state_value = value_optimizer.init(params_value)


    all_metrics = jnp.zeros((NUM_ITERS, 4))  # Assuming 4 metrics and 'num_iterations' training steps
    q_values = jnp.zeros((NUM_ITERS, 3, 3))
    v_values = jnp.zeros((NUM_ITERS, 1))

     # Initial training state
    train_state = (params_defender, params_attacker, params_value, opt_state_defender, opt_state_attacker, opt_state_value, rng_input, all_metrics, q_values, v_values)

    # Main training loop using jax.lax.fori_loop
    final_params_defender, final_params_attacker, final_params_value, final_opt_state_defender, final_opt_state_attacker, final_opt_state_value, final_rng_input, metrics, q_values, v_values = \
        jax.lax.fori_loop(0, NUM_ITERS, training_step, train_state)

    # Return the final parameters and optimizer states
    return final_params_defender, final_params_attacker, final_params_value, final_opt_state_defender, final_opt_state_attacker, final_opt_state_value, final_rng_input, metrics, q_values, v_values

    
 
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
folder = 'data/jax_nash/'+timestamp
#make folder
import os
if not os.path.exists(folder):
    os.makedirs(folder)




 

rng_input = jax.random.PRNGKey(1)
#actions_defender, actions_attacker, states,rewards, dones = rollout(rng_input, steps_in_episode)

env = ENV

obs, state ,nn_state = env.reset(rng_input)
state = env.encode_helper(state)
print(state.shape)


if is_train:
    output = train()

    params_defender, params_attacker, params_value, opt_state_defender, opt_state_attacker, opt_state_value, rng_input, metrics, q_values, v_values = output

    print(metrics)
    plt.plot(metrics[:,0])
    plt.savefig(folder+'/returns.png')
    plt.close()


    plt.plot(metrics[:,1])
    plt.savefig(folder+'/attacker_norm.png')
    plt.close()

    plt.plot(metrics[:,2])
    plt.savefig(folder+'/defender_norm.png')
    plt.close()

    plt.plot(metrics[:,3])
    plt.savefig(folder+'/value_loss.png')
    plt.close()

    df = pd.DataFrame(metrics)
    df.columns = ['returns', 'attacker_norm', 'defender_norm', 'value_loss']

    save_params(params_defender, params_attacker, params_value, folder)

    # rng_input = jax.random.PRNGKey(1125)

    #calculate bellman error
    
    
    bellman_error = [solve_nash(q,v) for q,v in zip(q_values, v_values)]
    plt.plot(bellman_error)
    plt.savefig(folder+'/bellman_error.png')
    plt.close()
    df['bellman_error'] = bellman_error
    df.to_csv(folder+'/all_metrics.csv')

        

else:
    with open('data/jax_nash/jax_nash_defender.pickle', 'rb') as handle:
        params_defender = pickle.load(handle)
    with open('data/jax_nash/jax_nash_attacker.pickle', 'rb') as handle:
        params_attacker = pickle.load(handle)

    with open('data/jax_nash/jax_nash_value.pickle', 'rb') as handle:
        params_value = pickle.load(handle)
    rng_input = jax.random.PRNGKey(91399)
    policy_net = hk.without_apply_rng(hk.transform(policy_network))






#do a rollout
policy_net = hk.without_apply_rng(hk.transform(policy_network))
value_net = hk.without_apply_rng(hk.transform(value_network))

obs, state, init_nn_state = env.reset(rng_input)

actions_defender, actions_attacker, states, nn_states, returns, dones, rewards = rollout(rng_input, state, init_nn_state, params_defender, params_attacker,STEPS_IN_EPISODE)
mask = jnp.logical_not(dones)

# print(states)




print('rendering')
states = convert_state_to_list(states)
env.render(states, dones)
env.make_gif(folder+'/rollout.gif')
# print(actions_defender)
# print(actions_attacker)
# print(dones)
# print(mask*returns)
# #rember to change seed
# # s back
# value_rollout = returns[0]
# q_matrix = get_q_matrix(rng_input, state, init_nn_state, params_defender, params_attacker, STEPS_IN_EPISODE)
# print(q_matrix)
# print('value_rollout', value_rollout)

# value = value_net.apply(params_value, init_nn_state)
# print('value network', value)

# print('bellman error', solve_nash(q_matrix, value_rollout))
# print('bellman errorvalue network', solve_nash(q_matrix, value))
# #rember to change seeds back