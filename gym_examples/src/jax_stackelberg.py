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
BATCH_SIZE = 1
NUM_INNER_ITERS = 1
NUM_ITERS = int(50000/NUM_INNER_ITERS)

#NUM_ITERS = int(40000/NUM_INNER_ITERS)


print('Steps in episode:', STEPS_IN_EPISODE)
print('Batch size:', BATCH_SIZE)
print('Number of inner iters:', NUM_INNER_ITERS)
print('Number of iters:', NUM_ITERS)
print('notes: added barrier condition')

is_train = True

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
        next_state, _, _, info = ENV.step(state, action,player)
        #get distance between players
        dist = np.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2])
        dists.append(dist)
    a = np.argmin(dists)
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


    def policy_step(state_input, _):
        """lax.scan compatible step transition in JAX environment."""
        state, nn_state, prev_done, rng = state_input
        rng, rng_step = jax.random.split(rng)
        action_defender = select_action(params_defender, policy_net, nn_state, rng_step)
        #action_defender = get_closer(state, 'defender')
        cur_state, cur_nn_state, reward, cur_done, g = env.step(state, action_defender, 'defender')
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
    actions_defender,actions_attacker, states, nn_states, rewards, dones, gs = scan_out
    mask = jnp.logical_not(dones)
    #update networks


    returns = calculate_discounted_returns(rewards, 0.99, mask)


    return actions_defender, actions_attacker, states, nn_states, returns, dones, rewards, gs









def save_params(defender_params, attacker_params, value_params, folder):
    with open(folder+'/jax_stack_defender.pickle', 'wb') as handle:
        pickle.dump(defender_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+'/jax_stack_attacker.pickle', 'wb') as handle:
        pickle.dump(attacker_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+'/jax_stack_value.pickle', 'wb') as handle:
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
            hk.Linear(2),
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
        params_attacker, params_defender, lamb,rng_input
    ):
        
        # _, init_state, init_nn_state = ENV.reset(rng_input)
            
        # _, _, _, _, returns, dones, _ = rollout(rng_input, init_state, init_nn_state, params_defender, params_attacker, 50)
        # padding_mask = jnp.logical_not(dones)

        # returns = padding_mask * returns
        # cum_rets = jnp.cumsum(returns)
        # #jax.debug.print("🤯 {x} 🤯", x=cum_rets)
        batched_rollout = jax.vmap(rollout, in_axes=(0,0,0, None, None, None), out_axes=0)

        keys = jax.random.split(rng_input, num=BATCH_SIZE)
        init_obs, initial_states, initial_nn_states = batched_env_reset(keys)

        _, _, _, _, all_returns, all_dones, _, all_gs = batched_rollout(keys,  initial_states, initial_nn_states, params_defender, params_attacker, STEPS_IN_EPISODE)
        all_masks =  jnp.logical_not(all_dones)
        mean_g = jnp.max(all_gs)


        
        returns = all_masks * (all_returns)
        cum_rets = jnp.cumsum(returns,axis=1)      

        return -(jnp.mean(cum_rets) - lamb*mean_g)

    @jax.jit
    def loss_defender(
        params_defender, params_attacker, lamb, rng_input
    ):
        _, init_state, init_nn_state = ENV.reset(rng_input)
            
        # _, _, _, _, returns, dones, _ = rollout(rng_input, init_state, init_nn_state, params_defender, params_attacker, 50)
        # padding_mask = jnp.logical_not(dones)

        batched_rollout = jax.vmap(rollout, in_axes=(0,0,0, None, None, None), out_axes=0)

        keys = jax.random.split(rng_input, num=BATCH_SIZE)
        init_obs, initial_states, initial_nn_states = batched_env_reset(keys)

        _, _, _, _, all_returns, all_dones, _, all_gs = batched_rollout(keys,  initial_states, initial_nn_states, params_defender, params_attacker, STEPS_IN_EPISODE)
        all_masks =  jnp.logical_not(all_dones)

        mean_g = jnp.max(all_gs)

        
        returns = all_masks * (all_returns - lamb* mean_g)
        cum_rets = jnp.cumsum(returns,axis=1)

        return jnp.mean(cum_rets) - lamb*mean_g
    

    @jax.jit
    def loss_lambda(
        lamb, params_defender, params_attacker, rng_input
    ):
        _, init_state, init_nn_state = ENV.reset(rng_input)
            
        # _, _, _, _, returns, dones, _ = rollout(rng_input, init_state, init_nn_state, params_defender, params_attacker, 50)
        # padding_mask = jnp.logical_not(dones)

        batched_rollout = jax.vmap(rollout, in_axes=(0,0,0, None, None, None), out_axes=0)

        keys = jax.random.split(rng_input, num=BATCH_SIZE)
        init_obs, initial_states, initial_nn_states = batched_env_reset(keys)

        _, _, _, _, all_returns, all_dones, _, all_gs = batched_rollout(keys,  initial_states, initial_nn_states, params_defender, params_attacker, STEPS_IN_EPISODE)
        all_masks =  jnp.logical_not(all_dones)
        mean_g = jnp.max(all_gs)


        
        returns = all_masks * (all_returns)
        cum_rets = jnp.cumsum(returns,axis=1) 

        return (jnp.mean(cum_rets) - lamb*mean_g)


# Define update function

    @jax.jit
    def update_defender(
        params_defender, params_attacker, opt_state, lamb, rng_input
    ):
        grads = jax.grad(loss_defender)(
            params_defender, params_attacker, lamb, rng_input
        )
       

        updates, opt_state = optimizer_defender.update(
            grads, params=params_defender, state=opt_state
        )

        norm = optax.global_norm(grads)
        return optax.apply_updates(params_defender, updates), opt_state, grads

    @jax.jit
    def update_attacker(
        params_attacker, params_defender, opt_state, lamb,rng_input
    ):
        grads = jax.grad(loss_attacker)(
            params_attacker, params_defender, lamb, rng_input
        )
        updates, opt_state = optimizer_attacker.update(
            grads, params=params_attacker, state=opt_state
        )

        norm = optax.global_norm(grads)
        return optax.apply_updates(params_attacker, updates), opt_state, grads
    
    @jax.jit
    def update_lambda(
        lamb, params_defender, params_attacker, opt_state, rng_input
    ):
        grads = jax.grad(loss_lambda)(
            lamb, params_defender, params_attacker, rng_input
        )
        updates, opt_state = optimizer_lamb.update(
            grads, params=params_attacker, state=opt_state
        )

        norm = optax.global_norm(grads)
        return optax.apply_updates(lamb, updates), opt_state, grads

    @jax.jit
    def value_loss(params_value, observations, returns, padding_mask):
        predicted_values = value_net.apply(params_value, observations).squeeze(-1)
        #predicted_values
        # Calculate MSE loss
        loss = jnp.mean(padding_mask * (predicted_values - returns) ** 2)
        return loss


    def batched_env_reset(rng_keys):
        # Vectorize the existing env.reset function to handle batched PRNG keys
        batched_reset = jax.vmap(env.reset)

        
        # Call the vectorized reset function with the batched PRNG keys
        init_obs, initial_states, nn_states = batched_reset(rng_keys)
        
        return init_obs, initial_states, nn_states
    

    def train_inner(i, train_state):
        """train inner player(attacker only)"""
        params_defender, params_attacker, lamb, opt_state_attacker, rng_input = train_state
        batched_rollout = jax.vmap(rollout, in_axes=(0,0,0, None, None, None), out_axes=0)
        keys = jax.random.split(rng_input, num=BATCH_SIZE)


        # Update the attacker network
        # Update the attacker network
        params_attacker, opt_state_attacker, attacker_grads = update_attacker(
            params_attacker, params_defender, opt_state_attacker, lamb, rng_input
        )

        return params_defender, params_attacker, lamb, opt_state_attacker, rng_input



    
    @loop_tqdm(NUM_ITERS)
    def training_outer(i, train_state):


        # Unpack the training state
        params_defender, params_attacker, lamb, opt_state_defender, opt_state_attacker, opt_state_lamb, rng_input, metrics, q_values, v_values  = train_state
        rng_input, subkey = jax.random.split(rng_input)

      


        batched_rollout = jax.vmap(rollout, in_axes=(0,0,0, None, None, None), out_axes=0)

        keys = jax.random.split(subkey, num=BATCH_SIZE)
        init_obs, initial_states, initial_nn_states = batched_env_reset(keys)

        all_actions_defender, all_actions_attacker, all_states, all_nn_states, all_returns, all_dones, all_rewards, all_gs = batched_rollout(keys,  initial_states, initial_nn_states, params_defender, params_attacker, STEPS_IN_EPISODE)

        all_masks = jnp.logical_not(all_dones)  
        mean_g = jnp.max(all_gs)

        
        returns = all_masks * (all_returns - lamb* mean_g)
        cum_rets = jnp.cumsum(returns,axis=1)



        # Update the attacker network (INNER UPDATE)
        inner_state = (params_defender, params_attacker, lamb, opt_state_attacker, rng_input)
        _, params_attacker, _, opt_state_attacker, _ = jax.lax.fori_loop(0, NUM_INNER_ITERS, train_inner, inner_state)

        # Update the defender network
        params_defender, opt_state_defender, defender_grads = update_defender(
            params_defender, params_attacker, opt_state_defender, lamb, subkey
        )
        
        # Update the value network
        params_value, opt_state_value, _ = update_lambda(
            lamb, params_defender, params_attacker, opt_state_lamb, subkey
        )

        #get bellman error
        #q, v = get_q_and_v2(rng_input, params_defender, params_attacker, params_value)


        #get metrics
        # Calculate the average return across all episodes
        average_return = jnp.mean(cum_rets) - lamb*mean_g
        #attacker_norm = optax.global_norm(attacker_grads)
        defender_norm = optax.global_norm(defender_grads)
        #bellman_error = jax.lax.stop_gradient(calc_nash_bellman_error(rng_input, params_defender, params_attacker))

        def_metric, att_matric = verify_equilibrium(params_defender, params_attacker, rng_input)

        m = (average_return, defender_norm, lamb, def_metric, att_matric)
        metrics = metrics.at[i, :].set(m)
        q_values = q_values.at[i, :].set(0)
        v_values = v_values.at[i, :].set(0)






        # Return updated training state
        return params_defender, params_attacker, params_value, opt_state_defender, opt_state_attacker, opt_state_value, subkey, metrics, q_values, v_values

    

    @jax.jit
    def evaluate_policy(params_defender, params_attacker, rng_input):
        batched_rollout = jax.vmap(rollout, in_axes=(0, 0, 0, None, None, None), out_axes=0)
        keys = jax.random.split(rng_input, num=BATCH_SIZE)
        init_obs, initial_states, initial_nn_states = batched_env_reset(keys)
        
        _, _, _, _, all_returns, all_dones, _, all_gs = batched_rollout(
            keys, initial_states, initial_nn_states, params_defender, params_attacker, STEPS_IN_EPISODE)
        
        all_masks =  jnp.logical_not(all_dones)
        mean_g = jnp.max(all_gs)


        
        returns = all_masks * (all_returns)
        cum_rets = jnp.cumsum(returns,axis=1) 
        
        # Calculate the expected return for the defender
        def_expected_return = (jnp.mean(cum_rets) - lamb*mean_g) / BATCH_SIZE
        
        # Calculate the expected return for the attacker
        att_expected_return = -(jnp.mean(cum_rets) - lamb*mean_g) / BATCH_SIZE
        
        return def_expected_return, att_expected_return

    @jax.jit
    def verify_equilibrium(params_defender, params_attacker, rng_input):
        # Evaluate the current policies
        def_expected_return, att_expected_return = evaluate_policy(params_defender, params_attacker, rng_input)
        
        # Evaluate the defender's policy against perturbed attacker's policies
        perturbed_attacker_params = jax.tree_map(lambda x: x + jax.random.normal(rng_input, x.shape) * 0.01, params_attacker)
        def_perturbed_returns = evaluate_policy(params_defender, perturbed_attacker_params, rng_input)[0]
        
        # Evaluate the attacker's policy against perturbed defender's policies
        perturbed_defender_params = jax.tree_map(lambda x: x + jax.random.normal(rng_input, x.shape) * 0.01, params_defender)
        att_perturbed_returns = evaluate_policy(perturbed_defender_params, params_attacker, rng_input)[1]
        
        # Check if the current policies are an equilibrium
        # is_def_best_response = jnp.all(def_expected_return >= def_perturbed_returns)
        # is_att_best_response = jnp.all(att_expected_return >= att_perturbed_returns)

        # Calculate the maximum difference between the current and perturbed returns for the defender
        def_diff = def_perturbed_returns - def_expected_return
        
        # Calculate the maximum difference between the current and perturbed returns for the attacker
        att_diff = att_perturbed_returns - att_expected_return
        
        # Calculate the equilibrium metric as the maximum of the defender and attacker differences
    
        return def_diff, att_diff
        

    
    env = ENV

    rng_input = jax.random.PRNGKey(4)
    #actions_defender, actions_attacker, states,rewards, dones = rollout(rng_input, steps_in_episode)


    obs, state ,nn_state = env.reset(rng_input)
    state = env.encode_helper(state)


    


    ############################
        # Initialize Haiku policy network
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    params_defender = policy_net.init(rng_input, nn_state)
    params_attacker = policy_net.init(rng_input, nn_state)

    lamb = 1000.0
    # Initialize Haiku optimizer
    optimizer_defender = optax.radam(1e-5, b1=0.9, b2=0.9)
    optimizer_attacker = optax.radam(1e-5, b1=0.9, b2=0.9)
    optimizer_lamb = optax.adam(1e-1)

    # Initialize optimizer state
    opt_state_attacker = optimizer_attacker.init(params_attacker)
    opt_state_defender = optimizer_defender.init(params_defender)
    opt_state_lamb= optimizer_lamb.init(lamb)


    all_metrics = jnp.zeros((NUM_ITERS, 5))  # Assuming 4 metrics and 'num_iterations' training steps
    q_values = jnp.zeros((NUM_ITERS, 3, 3))
    v_values = jnp.zeros((NUM_ITERS, 1))

     # Initial training state
    train_state = (params_defender, params_attacker, lamb, opt_state_defender, opt_state_attacker, opt_state_lamb, rng_input, all_metrics, q_values, v_values)

    # Main training loop using jax.lax.fori_loop
    final_params_defender, final_params_attacker, final_lamb, final_opt_state_defender, final_opt_state_attacker, final_opt_state_lamb, final_rng_input, metrics, q_values, v_values = \
        jax.lax.fori_loop(0, NUM_ITERS, training_outer, train_state)

    # Return the final parameters and optimizer states
    return final_params_defender, final_params_attacker, final_lamb, final_opt_state_defender, final_opt_state_attacker, final_opt_state_lamb, final_rng_input, metrics, q_values, v_values

    
 



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
folder = 'data/jax_cont_stack2/'+timestamp
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
    plt.savefig(folder+'/average_return.png')
    plt.close()


    # plt.plot(metrics[:,1])
    # plt.savefig('gifs/jax/attacker_norm_vmap_stack.png')
    # plt.close()

    plt.plot(metrics[:,1])
    plt.savefig(folder+'/defender_norm.png')
    plt.close()

    plt.plot(metrics[:,2])
    plt.savefig(folder+'/lambda.png')
    plt.close()

    plt.plot(metrics[:,3])
    plt.savefig(folder+'/def_metric.png')
    plt.close()

    plt.plot(metrics[:,4])
    plt.savefig(folder+'/att_metric.png')
    plt.close()

    save_params(params_defender, params_attacker, params_value, folder)
    rng_input = jax.random.PRNGKey(1125)
    df = pd.DataFrame(metrics)
    df.columns = ['average_return', 'defender_norm', 'lambda', 'def_metric', 'att_metric']

    #calculate bellman error
    
    
    # bellman_error = [solve_stackelberg(q.T,v) for q,v in zip(q_values, v_values)]
    # plt.plot(bellman_error)
    # plt.savefig(folder+'/bellman_error.png')
    # plt.close()
    # df['bellman_error'] = bellman_error
    df.to_csv(folder+'/all_metrics.csv')


    
        

else:
    #actions_defender, actions_attacker, states,rewards, dones = rollout(rng_input, steps_in_episode)


    # with open('data/jax_stack/jax_stack_defender.pickle', 'rb') as handle:
    #     params_defender = pickle.load(handle)
    # with open('data/jax_stack/jax_stack_attacker.pickle', 'rb') as handle:
    #     params_attacker = pickle.load(handle)
    # with open('data/jax_stack/jax_stack_value.pickle', 'rb') as handle:
    #     params_value = pickle.load(handle)
    # #rng_input = jax.random.PRNGKey(69679898967)
    # rng_input = jax.random.PRNGKey(114258)
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    params_defender = policy_net.init(rng_input, nn_state)
    params_attacker = policy_net.init(rng_input, nn_state)




    

   


    




#do a rollout
policy_net = hk.without_apply_rng(hk.transform(policy_network))
value_net = hk.without_apply_rng(hk.transform(value_network))

obs, state, init_nn_state = env.reset(rng_input)


actions_defender, actions_attacker, states, nn_states, returns, dones, rewards, gs = rollout(rng_input, state, init_nn_state, params_defender, params_attacker,STEPS_IN_EPISODE)
mask = jnp.logical_not(dones)
#set mask to all 1
#dones = jnp.zeros_like(dones)


# print(states)




print('rendering')
states = convert_state_to_list(states)
env.render(states, dones)
env.make_gif(folder+'/rollout.gif')
# print(actions_defender)
# print(actions_attacker)
# print(dones)
print(mask*returns)
print(gs)


# value_rollout = returns[0]
# q_matrix = get_q_matrix(rng_input, state, init_nn_state, params_defender, params_attacker, STEPS_IN_EPISODE)
# print(q_matrix)
# print('value_rollout', value_rollout)

# value = value_net.apply(params_value, init_nn_state)
# print(value)

# print('bellman error', solve_nash(q_matrix, value_rollout))
# print('bellman errorvalue network', solve_nash(q_matrix, value))


# print('stackelberg rollout', solve_stackelberg(q_matrix.T, value_rollout))
# print('stackelberg', solve_stackelberg(q_matrix.T, value))

# q, v = get_q_and_v(rng_input, params_defender, params_attacker)
# bellman_error = solve_stackelberg(q.T, v)
# print('bellman error', bellman_error)
# #rember to change seeds back
# print('***********')

# # q, v = get_q_and_v2(rng_input, params_defender, params_attacker, params_value)
# # print(q)
# # print(v)
# # bellman_error = solve_stackelberg(q.T, v)


# # print('bellman error2', bellman_error)