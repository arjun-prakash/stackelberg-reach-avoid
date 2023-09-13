# Implement the REINFORCE algorithm
import multiprocessing as mp
import sys
from collections import Counter
from pprint import pprint

from pathos.pools import ProcessPool
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
from envs.two_player_dubins_car import TwoPlayerDubinsCarEnv
from matplotlib import cm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import yaml


def parallel_rollouts(
    env,
    params,
    policy_net,
    num_rollouts,
    key,
    epsilon,
    gamma,
    render=False,
    for_q_value=False,
):
    keys = jax.random.split(key, num_rollouts)
    args = [
        (env.game_type, params, policy_net, k, epsilon, gamma, render, for_q_value)
        for k in keys
    ]
    with ProcessPool() as pool:
        results = pool.map(env.single_rollout, args)
    (
        all_states,
        all_actions,
        all_action_masks,
        all_returns,
        all_masks,
        all_wins,
    ) = zip(*results)
    return (
        all_states,
        all_actions,
        all_action_masks,
        all_returns,
        all_masks,
        all_wins,
    )

def get_values(env, params, policy_net, grid_state, num_rollouts, key, epsilon, gamma):
        initial_state = env.set(grid_state[0], grid_state[1], grid_state[2], grid_state[3], grid_state[4], grid_state[5])
        #print('initial_state values', env.state)
        states, actions, action_masks, returns, masks, wins = parallel_rollouts(
            env,
            params,
            policy_net,
            num_rollouts,
            key,
            epsilon,
            gamma,
            render=False,
            for_q_value=False,
        )
        attacker_returns = [r["attacker"][0] for r in returns]
        v = np.mean(attacker_returns)
        return v

def get_q_values(env, params, policy_net, grid_state, num_rollouts, key, epsilon, gamma):

    move_list = []
    for d_action in range(env.action_space["defender"].n):
        initial_state = env.set(grid_state[0], grid_state[1], grid_state[2], grid_state[3], grid_state[4], grid_state[5])
        #print('initial_state q_values', initial_state)
        defender_state, d_reward, _, _ = env.step(
            initial_state, d_action, "defender", update_env=True
        )
        for a_action in range(env.action_space["attacker"].n):
            state, reward, _, _ = env.step(
                defender_state, a_action, "attacker", update_env=True
            )
            #print('state after step', state)
            states, actions, action_masks, returns, masks, wins = parallel_rollouts(
                env,
                params,
                policy_net,
                num_rollouts,
                key,
                epsilon,
                gamma,
                render=False,
                for_q_value=True,
            )
            attacker_returns = [r["attacker"][0] for r in returns]
            mean_attacker_returns = np.mean(attacker_returns)
            move_list.append(
                {
                    "defender": d_action,
                    "attacker": a_action,
                    "q_value": mean_attacker_returns,
                }
            )
    print(move_list)
    q_values = np.array([move["q_value"] for move in move_list]).reshape(
        env.num_actions, env.num_actions
    )
    best_attacker_moves = np.argmax(q_values, axis=1)
    best_defender_move = np.argmin(np.max(q_values, axis=1))
    best_attacker_move = best_attacker_moves[best_defender_move]
    q = q_values[best_defender_move][best_attacker_move]
    return q

def calc_bellman_error(env, params, policy_net, num_rollouts, key, epsilon, gamma):
    x_a = np.linspace(-2, 2, 4)
    y_a = np.linspace(2, 2, 1)
    theta_a = np.linspace(0, 0, 1)
    x_d = np.linspace(-1, 1, 2)
    y_d = np.linspace(0, 0, 1)
    theta_d = np.linspace(0, 0, 1)

    xxa, yya, tta, xxd, yyd, ttd = np.meshgrid(x_a, y_a, theta_a, x_d, y_d, theta_d)
    grid = np.vstack(
        [
            xxa.ravel(),
            yya.ravel(),
            tta.ravel(),
            xxd.ravel(),
            yyd.ravel(),
            ttd.ravel(),
        ]
    ).T

    v_vector = []
    q_vector = []
    for g in tqdm(grid):
        v = get_values(
            env, params, policy_net, g, num_rollouts, jax.random.PRNGKey(42), 0.01, 0.95
        )
        v_vector.append(v)

        q = get_q_values(
            env, params, policy_net, g, num_rollouts, jax.random.PRNGKey(42), 0.01, 0.95
        )
        q_vector.append(q)

    return np.array(q_vector), np.array(v_vector)#np.linalg.norm(np.array(q_vector) - np.array(v_vector))


def parallel_nash_reinforce(
    env,
    num_episodes,
    learning_rate,
    gamma,
    batch_multiple,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    num_parallel,
    eval_interval,
    num_eval_episodes,
    writer,
    timestamp,
    params_list,
):
    # Define loss function
    # Define loss function
    
    

    


    def policy_network(observation):
        net = hk.Sequential(
            [
                hk.Linear(100),
                jax.nn.leaky_relu,
                hk.Linear(100),
                jax.nn.leaky_relu,
                hk.Linear(100),
                jax.nn.leaky_relu,
                hk.Linear(100),
                jax.nn.leaky_relu,
                hk.Linear(env.num_actions),
                jax.nn.softmax,
            ]
        )
        return net(observation)

    ############################
    # Initialize Haiku policy network
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    initial_state = env.reset()

    for file in params_list:
        with open(file, 'rb') as handle:
            loaded_params = pickle.load(handle)
            episode = int(file.split('_episode_')[1].split('_params')[0])
            bellman_error = calc_bellman_error(env, loaded_params, policy_net, num_eval_episodes, jax.random.PRNGKey(episode), epsilon_start, gamma)
            print('bellman_error', bellman_error)
            #writer.add_scalar('bellman_error', bellman_error, episode)

        


def parallel_stackelberg_reinforce(
    env,
    num_episodes,
    learning_rate,
    gamma,
    batch_multiple,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    num_parallel,
    eval_interval,
    num_eval_episodes,
    writer,
    timestamp,
    params_list,
):
    # Define loss function
    # Define loss function
    

   

    

    def policy_network(observation, legal_moves):
        net = hk.Sequential(
            [
                hk.Linear(100),
                jax.nn.relu,
                hk.Linear(100),
                jax.nn.relu,
                hk.Linear(100),
                jax.nn.relu,
                hk.Linear(100),
                jax.nn.relu,
                hk.Linear(env.num_actions),
                jax.nn.softmax,
            ]
        )
        # legal_moves = legal_moves[..., None]

        logits = net(observation)
        # legal_moves = jnp.broadcast_to(legal_moves, logits.shape)  # Broadcast to the shape of logits

        # masked_logits = jnp.multiply(logits, legal_moves)
        masked_logits = jnp.where(legal_moves, logits, 1e-8)

        # probabilities = jax.nn.softmax(masked_logits)
        return masked_logits

    ############################
    # Initialize Haiku policy network
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    initial_state = env.reset()

    for file in params_list:
        with open(file, 'rb') as handle:
            loaded_params = pickle.load(handle)
            episode = int(file.split('_episode_')[1].split('_params')[0])
            q,v = calc_bellman_error(env, loaded_params, policy_net, num_eval_episodes, jax.random.PRNGKey(episode), epsilon_start, gamma)
            q_norm = np.linalg.norm(q)
            v_norm = np.linalg.norm(v)
            bellman_error = np.linalg.norm(q-v)
            print('q:', q)
            print('v:', v)
            print('bellman_error:', bellman_error)
            # writer.add_scalar('q_norm', q_norm, episode)
            # writer.add_scalar('v_norm', v_norm, episode)
            # writer.add_scalar('bellman_error', bellman_error, episode)

            writer.add_scalars(
                "values",
                {
                    "q_norm": q_norm,
                    "v_norm": v_norm,
                    "bellman": bellman_error,
                },
                episode,
            )   


      




def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def print_config(config):
    print("Starting experiment with the following configuration:\n")
    print(yaml.dump(config))





if __name__ == "__main__":
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

    # Hyperparameters
    learning_rate = config['hyperparameters']['learning_rate']
    gamma = config['hyperparameters']['gamma']
    epsilon_start = config['hyperparameters']['epsilon_start']
    epsilon_end = config['hyperparameters']['epsilon_end']
    epsilon_decay = config['hyperparameters']['epsilon_decay']

    # Training parameters
    batch_multiple = config['training']['batch_multiple']  # the batch size will be num_parallel * batch_multiple
    num_episodes = config['training']['num_episodes']
    loaded_params = config['training']['loaded_params']
    num_parallel = config['training']['num_parallel'] #mp.cpu_count()
    if num_parallel != config['training']['num_parallel']: 
        ValueError("num_parallel in config file does not match the number of cores on this machine")
    print("cpu_count", num_parallel, "config", config['training']['num_parallel'])

    # Evaluation parameters
    eval_interval = config['eval']['eval_interval']
    num_eval_episodes = config['eval']['num_eval_episodes']

    # Logging
    print(game_type, " starting experiment at :", timestamp)
    writer = SummaryWriter(f"runs3/experiment_{game_type}" + timestamp+"_bellman_error_fix")

    import glob
    import pickle

    # Get a list of all files in the directory
    #files = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_nash/2023-08-24 15:30:06.415206_episode_*_params.pickle')
    #files = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_stackelberg/2023-08-23 13:33:43.910321_episode_*_params.pickle')
    files = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_stackelberg/2023-09-11 13:26:38.127129_episode_*_params.pickle')

    files.sort(key=lambda x: int(x.split('_episode_')[1].split('_params')[0]))
    files = files[::2]
    print(files)

    # Now params_list contains the parameters from all episodes

    # Training
    if game_type == "nash":
        trained_params = parallel_nash_reinforce(
            env,
            num_episodes=num_episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_multiple=batch_multiple,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            num_parallel=num_parallel,
            eval_interval=eval_interval,
            num_eval_episodes=num_eval_episodes,
            writer=writer,
            timestamp=timestamp,
            params_list=files,
        )
    elif game_type == "stackelberg":
        trained_params = parallel_stackelberg_reinforce(
            env,
            num_episodes=num_episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_multiple=batch_multiple,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            num_parallel=num_parallel,
            eval_interval=eval_interval,
            num_eval_episodes=num_eval_episodes,
            writer=writer,
            timestamp=timestamp,
            params_list=files,
        )


#don't have shedule for epsilon decay, just fix it.  upghrade learning rate

# measure the right error, and make sure it goes down. 
# bayseian persuasion, principle agent 