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
import matplotlib.pyplot as plt
from matplotlib import cm

from torch.utils.tensorboard import SummaryWriter

import imageio
from PIL import Image
import datetime
import pickle



# Define the policy network
def policy_network(observation):
    net = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(env.num_actions),
        jax.nn.softmax
    ])
    return net(observation)



def test_policy(env, params, policy_net, state=None, make_gif=True, filename='episode.gif'):
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    key = jax.random.PRNGKey(1)

    if state is None:
        state = env.reset()
    #state = env.set(-2., 0., 0., 3., 0., 0.)
    #state = env.reset()
    nn_state = env.encode_helper(state)
    frames = []
    done = False
    episode_length = 0
    attacker_rewards = []

    while not done and episode_length < 200:
        for player in env.players:
        
            key, subkey = jax.random.split(key)

            #legal_actions_mask = env.get_legal_actions_mask(state, player)

            probs = policy_net.apply(params[player], nn_state)

            
            action = jax.random.categorical(subkey, probs)
            #action = np.argmax(probs)
            #print(reweighted_probs)
            state, rewards, done, info = env.step(state=state, action=action, player=player, update_env=True)
            if player == 'attacker': attacker_rewards.append(rewards)

            nn_state = env.encode_helper(state)
            episode_length += 1
            if make_gif: env.render()

    if make_gif: env.make_gif("two_player_animation_nash_pg.gif")

    return attacker_rewards
    


def save_params(episode_num, params):
    with open(f'data/nash_{episode_num}_params.pickle', 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


from tqdm import tqdm

def calculate_returns(rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    return returns[::-1]  # reverse the list

def get_values(grid, trained_params, policy_net, num_episodes=2):
    values= []
    for i in tqdm(range(len(grid))):
        v = []
        for e in range(num_episodes):
            state = env.set(grid[i][0], grid[i][1], grid[i][2])
            episode_rewards = test_policy(env, trained_params, policy_net, state=state, make_gif=False)
            returns = calculate_returns(episode_rewards)
            v.append(returns[0])



        values.append(np.mean(v))
    return np.array(values)


def get_q_values(grid, trained_params, policy_net, num_episodes=2):
    max_q = []
    for i in tqdm(range(len(grid))):
        q_values =[] #the q values for each pair of actions
        for e in range(num_episodes):
            q_values_episode =[] 
            possible_actions_d = [] #track defender actions
            for d_action in range(env.action_space['defender'].n):
                state = env.set(grid[i][0], grid[i][1], grid[i][2]) #set the state after the defender move
                defender_state ,d_reward,_, _ = env.step(i, d_action, 'defender', update_env=True)
                possible_actions_a = [] #track the attacker actions

                for a_action in range(env.action_space['attacker'].n):
                    #set the reset the state back to the defender move
                    state = env.set(defender_state['attacker'][0], defender_state['attacker'][1], defender_state['attacker'][2], 
                    defender_state['defender'][0], defender_state['defender'][1], defender_state['defender'][2])

                    state,a_reward,_, _ = env.step(i, a_action, 'attacker', update_env=True) #step through the attacker move
                    episode_rewards = [a_reward] + test_policy(env, trained_params, policy_net, state, make_gif=False) #rollout the trajectory
                    returns = calculate_returns(episode_rewards)
                    possible_actions_a.append(returns[0]) #append the reward of the state

                possible_actions_d.append(possible_actions_a) #append the list of attacker responses to the defender action
            q_values_episode.append(possible_actions_d) #append the list of defender, attacker pairs
        q_values.append(np.mean(q_values_episode,axis=0)) #append the mean of the q values for each pair of actions

        q_values = np.array(q_values).reshape(3,3)
        best_attacker_moves = np.argmax(q_values,axis=1)
        best_defender_move =  np.argmin(np.max(q_values,axis=1))
        best_attacker_move = best_attacker_moves[best_defender_move]
        best_value = q_values[best_defender_move][best_attacker_move]
        max_q.append(best_value)

    return np.array(max_q)


def calculate_bellman_error(grid, trained_params, policy_net, num_episodes=10):
    return np.max(get_values(grid, trained_params, policy_net,num_episodes) - get_q_values(grid, trained_params, policy_net,num_episodes))


def make_grid():
    x_s = np.linspace(-4,4, 3)
    y_s = np.linspace(-4,4, 3)
    theta_s = np.linspace(0,2*np.pi, 3)

    xx, yy, tt = np.meshgrid(x_s, y_s, theta_s)
    grid = np.vstack([xx.ravel(), yy.ravel(), tt.ravel()]).T
    return grid

# Implement the REINFORCE algorithm
def reinforce_nash(env, num_episodes, learning_rate, gamma, batch_size, epsilon_start, epsilon_end, epsilon_decay):
    # Initialize Haiku policy network
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    initial_state = env.reset()
    initial_state_nn = env.encode_helper(initial_state)

    params = {player: policy_net.init(jax.random.PRNGKey(42), initial_state_nn) for player in env.players}
    grid = make_grid()

    # Define the optimizer
    agent_optimizer = optax.chain(
          optax.clip(1.0),
          optax.adam(learning_rate=learning_rate)
        )

    optimizer = {player: agent_optimizer for player in env.players}  
    
    opt_state = {player: optimizer[player].init(params[player]) for player in env.players}

    # Define loss function
    @jax.jit
    def loss_attacker(params, observations, actions, returns): #apply no grad to returns maybe
        action_probabilities = policy_net.apply(params, observations)
        log_probs = jnp.log(jnp.take_along_axis(action_probabilities, actions[..., None], axis=-1))
        return -jnp.mean(log_probs * jax.lax.stop_gradient(returns)) #Maybe

    @jax.jit
    def loss_defender(params, observations, actions, returns): #apply no grad to returns maybe
        action_probabilities = policy_net.apply(params, observations)
        log_probs = jnp.log(jnp.take_along_axis(action_probabilities, actions[..., None], axis=-1))
        return jnp.mean(log_probs * jax.lax.stop_gradient(returns)) #Maybe

    # Define update function
    
    @jax.jit
    def update_defender(params, opt_state, observations, actions, returns):
        grads = jax.grad(loss_defender)(params, observations, actions, returns)
        updates, opt_state = optimizer['defender'].update(grads, params=params, state=opt_state)
        return optax.apply_updates(params, updates), opt_state
    
    @jax.jit
    def update_attacker(params, opt_state, observations, actions, returns):
        grads = jax.grad(loss_attacker)(params, observations, actions, returns)
        updates, opt_state = optimizer['attacker'].update(grads, params=params, state=opt_state)
        return optax.apply_updates(params, updates), opt_state
    
    def select_action(nn_state, params, key, epsilon):
        if jax.random.uniform(key) < epsilon:
            return jax.random.choice(key, env.num_actions)
        else:
            probs = policy_net.apply(params, nn_state)
            return jax.random.categorical(key, probs)

    episode_losses = []  # Add this line to store the losses
    all_wins, all_traj_lengths = [], []
    # Train the policy network
    # Train the policy network
    batch_states = {player: [] for player in env.players}
    batch_actions = {player: [] for player in env.players}
    batch_returns = {player: [] for player in env.players}
    batch_wins = {player: [] for player in env.players}

    traj_length = []

    epsilon = epsilon_start

    for episode in range(num_episodes):

        
        key = jax.random.PRNGKey(episode)
        valid = True
        render = False
        state = env.reset()
        nn_state = env.encode_helper(state)
        
        states = {player: [] for player in env.players}
        actions = {player: [] for player in env.players}
        rewards = {player: [] for player in env.players}
        wins = {player: 0 for player in env.players+['draw']}

        episode_length = 0
        print(f"Episode {episode} started...")
        
        
        done = False
        valid=True
        defender_wins = False
        attacker_wins = False

        
        if (episode) % 500 == 0: 
            save=True
        else:
            save=False


        while not done and not defender_wins and not attacker_wins and episode_length < 200:
            
            for player in env.players:            

                states[player].append(nn_state)
                key, subkey = jax.random.split(key)
                
                action = select_action(nn_state, params[player], subkey, epsilon)  # Use the new function here

                state, reward, done, info = env.step(state=state, action=action, player=player, update_env=True)
                nn_state = env.encode_helper(state)
                actions[player].append(action)
                rewards[player].append(reward)
                episode_length += 1

                if done and player == 'defender': #only attacker can end the game, iterate one more time
                    defender_wins = True
                    wins['defender'] += 1

                if (defender_wins and player == 'attacker'): #overwrite the attacker's last reward
                    rewards['attacker'][-1] = -10
                    done = True
                    break

                if (done and player == 'attacker'): #break if attacker wins, game is over
                    if info['is_legal']:
                        attacker_wins = True
                        wins['attacker'] += 1
                    elif not info['is_legal']:
                        defender_wins = True
                        wins['defender'] += 1
                    break
                if save:
                    env.render()
                    

            

            # if jax.random.uniform(subkey) < 1 - gamma:
            #     print('game ended by gamma after ', episode_length, ' steps')
            #     break
                
            traj_length.append(episode_length)

        if save:
            env.make_gif(f'gifs/nash_script/{episode}.gif')
            

        if not attacker_wins and not defender_wins: #if defender wins, done would actually be false
            wins['draw'] += 1

        
                



        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        returns = {player: [] for player in env.players}

        for player in env.players:
            G = 0
            for r in reversed(rewards['attacker']):
                G = r + gamma * G
                returns[player].append(G)
            returns[player] = list(reversed(returns[player]))
                
            batch_states[player].extend(states[player])
            batch_actions[player].extend(actions[player])
            batch_returns[player].extend(returns[player])
            batch_wins[player].append(wins[player])

        if (episode + 1) % batch_size == 0 and valid:
            for player in env.players:
            

                # Normalize the returns for the entire batch
                # Update the policy parameters using the batch of episodes
                
                #params[player], opt_state[player] = update(params[player], opt_state[player], np.array(batch_states[player]), np.array(batch_actions[player]), np.array(batch_returns[player]), player )
                
                if player == 'defender':
                    params[player], opt_state[player] = update_defender(params[player], opt_state[player], np.array(batch_states[player]), np.array(batch_actions[player]), np.array(batch_returns[player]))
                elif player == 'attacker':
                    params[player], opt_state[player] = update_attacker(params[player], opt_state[player], np.array(batch_states[player]), np.array(batch_actions[player]), np.array(batch_returns[player]))

                
                print(f"Episode {episode} finished", 'returns:', np.mean(batch_returns[player]))
                print('num wins', wins)
                print('average_length', np.mean(traj_length))

                # writer.add_scalar('loss', np.array(loss_value), episode)
                # writer.add_scalar('traj length', np.array(np.mean(traj_length)), episode)
                #writer.add_scalar('attacker_returns', np.array(np.mean(batch_returns['attacker'])), episode)

                writer.add_scalars('returns', {'defender': np.array(np.mean(batch_returns['defender'])),
                               'attacker': np.array(np.mean(batch_returns['attacker']))},
                   episode)
                
                writer.add_scalars('num_wins', {'defender':np.array(wins['defender']),'attacker':np.array(wins['attacker']), 'draw': np.array(wins['draw'])}, episode)





            all_wins.append(wins)
            all_traj_lengths.append(np.mean(traj_length))
            # Clear the batch
            batch_states = {player: [] for player in env.players}
            batch_actions = {player: [] for player in env.players}
            batch_returns = {player: [] for player in env.players}
            batch_wins = {player: [] for player in env.players}
            wins = 0
            traj_length = []

        if save:
            save_params(episode, params)
            bellman_error = calculate_bellman_error(grid,params, policy_network, num_episodes=2)
            SummaryWriter('bellman_error', bellman_error, episode)
                
            
    return params





if __name__ == "__main__":
    # Train the agent using REINFORCE algorithm
    writer = SummaryWriter(f'runs/nash_script_' + str(datetime.datetime.now()))

    env = TwoPlayerDubinsCarEnv()

    learning_rate = 1e-6#1e-3
    gamma = 0.99
    batch_size = 1
    num_episodes = 600



    trained_params = reinforce_nash(env, num_episodes=num_episodes, learning_rate=learning_rate, 
                                    gamma=gamma, batch_size=batch_size,
                                    epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99)
