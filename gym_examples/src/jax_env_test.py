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
#from torch.utils.tensorboard import SummaryWriter
import yaml
from functools import partial
from jax_tqdm import scan_tqdm, loop_tqdm

from jax import config
config.update('jax_platform_name', 'gpu')





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

    def policy_step(state_input, _):
        """lax.scan compatible step transition in JAX environment."""
        state, nn_state, prev_done, rng = state_input
        rng, rng_step = jax.random.split(rng)
        #action_defender = random_policy(rng_step)
        action_defender = select_action(params_defender, policy_net, nn_state, rng_step)
        next_state, nn_state, reward, next_done = env.step(rng_step, state, action_defender, 'defender')
        #action_attacker = random_policy(rng_step)
        action_attacker =select_action(params_attacker, policy_net, nn_state, rng_step)
        next_state, nn_state, reward, next_done = env.step(rng_step, next_state, action_attacker, 'attacker')
        done = jnp.logical_or(prev_done, next_done)
        carry = (next_state,nn_state, done, rng_step)


        return carry, (action_defender, action_attacker, next_state, nn_state,reward, done)
    
    def calculate_discounted_returns(rewards, discount_factor):
        """Calculate discounted cumulative returns using JAX vectorized operations."""
        # Reverse the rewards array
        rewards_reversed = jnp.flip(rewards, axis=0)
        
        # Calculate discounted returns
        discounts = discount_factor ** jnp.arange(rewards_reversed.size)
        discounted_rewards_reversed = rewards_reversed * discounts
        
        # Compute the cumulative sum in reverse order and then reverse it back
        discounted_returns = jnp.flip(jnp.cumsum(discounted_rewards_reversed), axis=0)

        return discounted_returns



    env = TwoPlayerDubinsCarEnv(
        game_type=game_type,
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

    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    value_net = hk.without_apply_rng(hk.transform(value_network))


    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state, nn_state = env.reset(rng_reset)

    


    # Scan over the episode step loop
    initial_carry = (state, nn_state, False, rng_episode)
    _, scan_out = jax.lax.scan(policy_step, initial_carry, None, length=steps_in_episode)

    # Unpack scan output
    actions_defender,actions_attacker, states, nn_states, rewards, dones = scan_out

    #update networks

    returns = calculate_discounted_returns(rewards, 0.99)


    return actions_defender, actions_attacker, states, nn_states, returns, dones

def rollout_body(i, carry):
    rng_input,  params_defender, params_attacker, params_value, all_actions_defender, all_actions_attacker ,all_nn_states, all_returns, all_dones = carry
    rng_input, subkey = jax.random.split(rng_input)
    actions_defender, actions_attacker, states, nn_states, returns, dones = rollout(subkey, params_defender, params_attacker, 50)

    # Store the results of this rollout
    all_actions_defender = all_actions_defender.at[i, :].set(actions_defender)
    all_actions_attacker = all_actions_attacker.at[i, :].set(actions_attacker)
    all_nn_states = all_nn_states.at[i, :].set(nn_states) 
    all_returns = all_returns.at[i, :].set(returns)
    all_dones = all_dones.at[i, :].set(dones)

    return subkey, params_defender, params_attacker, params_value, all_actions_defender, all_actions_attacker, all_nn_states, all_returns, all_dones











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
        baseline_values = value_net.apply(value_params, observations)
        advantage = returns - baseline_values.squeeze(-1)

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
        updates, opt_state = optimizer_defender.update(
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
    
    @loop_tqdm(1000)
    def training_step(i, train_state):


        # Unpack the training state
        params_defender, params_attacker, params_value, opt_state_defender, opt_state_attacker, opt_state_value, rng_input, metrics = train_state
        rng_input, subkey = jax.random.split(rng_input)

        # Adjust the shape based on the actual structure of your state and action
        all_actions_defender = jnp.zeros((num_episodes, steps_in_episode), dtype=jnp.int32)
        all_actions_attacker = jnp.zeros((num_episodes, steps_in_episode), dtype=jnp.int32)
        all_states = jnp.zeros((num_episodes, steps_in_episode,3))  # Add the shape of your state
        all_nn_states = jnp.zeros((num_episodes, steps_in_episode,13))  # Add the shape of your state
        all_returns = jnp.zeros((num_episodes, steps_in_episode))
        all_dones = jnp.zeros((num_episodes, steps_in_episode), dtype=jnp.bool_)
        all_metrics = jnp.zeros((num_episodes, 4))

        # Perform rollouts and collect data
        final_rng_input, params_defender, params_attacker, params_value, all_actions_defender, all_actions_attacker, all_nn_states, all_returns, all_dones = \
            jax.lax.fori_loop(
                0, num_episodes, 
                rollout_body, 
                (subkey, params_defender, params_attacker, params_value, all_actions_defender, all_actions_attacker, all_nn_states, all_returns, all_dones)
            )
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


        #get metrics
        # Calculate the average return across all episodes
        average_return = jnp.mean(jnp.sum(all_returns, axis=1))
        attacker_norm = optax.global_norm(params_attacker)
        defender_norm = optax.global_norm(params_defender)
        current_value_loss = value_loss(params_value, all_nn_states, all_returns, all_masks)

        m = (average_return, attacker_norm, defender_norm, current_value_loss)
        metrics = metrics.at[i, :].set(m)




        # Return updated training state
        return params_defender, params_attacker, params_value, opt_state_defender, opt_state_attacker, opt_state_value, subkey, metrics

    


    
    env = TwoPlayerDubinsCarEnv(
            game_type=game_type,
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


    rng_input = jax.random.PRNGKey(0)
    #actions_defender, actions_attacker, states,rewards, dones = rollout(rng_input, steps_in_episode)
    steps_in_episode = 50
    num_episodes = 32


    steps_in_episode = 50

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
    optimizer_defender = optax.radam(1e-5, b1=0.9, b2=0.9)
    optimizer_attacker = optax.radam(1e-5, b1=0.9, b2=0.9)
    value_optimizer = optax.radam(1e-5, b1=0.9, b2=0.9)

    # Initialize optimizer state
    opt_state_attacker = optimizer_attacker.init(params_attacker)
    opt_state_defender = optimizer_defender.init(params_defender)
    opt_state_value = value_optimizer.init(params_value)

    num_iters = 1000

    all_metrics = jnp.zeros((num_iters, 4))  # Assuming 4 metrics and 'num_iterations' training steps

     # Initial training state
    train_state = (params_defender, params_attacker, params_value, opt_state_defender, opt_state_attacker, opt_state_value, rng_input, all_metrics )

    # Main training loop using jax.lax.fori_loop
    final_params_defender, final_params_attacker, final_params_value, final_opt_state_defender, final_opt_state_attacker, final_opt_state_value, final_rng_input, metrics = \
        jax.lax.fori_loop(0, num_iters, training_step, train_state)

    # Return the final parameters and optimizer states
    return final_params_defender, final_params_attacker, final_params_value, final_opt_state_defender, final_opt_state_attacker, final_opt_state_value, final_rng_input, metrics

    
 





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
        goal_position=[0,-3],
        goal_radius=1,
        timestep=1,
        v_max=0.25,
        omega_max=30,
    )   

 

rng_input = jax.random.PRNGKey(1)
steps_in_episode = 50
#actions_defender, actions_attacker, states,rewards, dones = rollout(rng_input, steps_in_episode)
steps_in_episode = 50
num_episodes = 100


steps_in_episode = 50

obs, state ,nn_state = env.reset(rng_input)
state = env.encode_helper(state)
print(state.shape)


output = train()

params_defender, params_attacker, params_value, opt_state_defender, opt_state_attacker, opt_state_value, rng_input, metrics = output

#do a rollout
rng_input = jax.random.PRNGKey(1125)
steps_in_episode = 50
policy_net = hk.without_apply_rng(hk.transform(policy_network))
value_net = hk.without_apply_rng(hk.transform(value_network))



actions_defender, actions_attacker, states, nn_states, returns, dones= rollout(rng_input, params_defender, params_attacker, steps_in_episode)
#verage_return, attacker_norm, defender_norm, current_value_loss
print(metrics)
plt.plot(metrics[:,0])
plt.savefig('gifs/jax/met1.png')




print('rendering')
# print(states)
states = convert_state_to_list(states)
env.render(states, dones)
env.make_gif('gifs/jax/test_train.gif')

