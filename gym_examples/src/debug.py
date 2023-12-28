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
from envs.two_player_grid_world import TwoPlayerGridWorldEnv
from matplotlib import cm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import yaml
import copy
import itertools

import cvxpy as cp


def solve_stackelberg_game(q_values_data):
    # Convert q_values_data to matrix
    num_defender_actions = max([entry['defender'] for entry in q_values_data]) + 1
    num_attacker_actions = max([entry['attacker'] for entry in q_values_data]) + 1
    Q_matrix = np.zeros((num_attacker_actions, num_defender_actions))
    for entry in q_values_data:
        Q_matrix[entry['attacker']][entry['defender']] = entry['q_value']
    print(Q_matrix)
    # Variables
    q = cp.Variable(Q_matrix.shape[1], nonneg=True)  # Defender's strategy
    z = cp.Variable()  # Worst-case expected payoff for defender

    # Objective: Minimize z (worst-case expected payoff for defender)
    objective = cp.Minimize(z)

    # Constraints
    constraints = [
        cp.sum(q) == 1,  # Defender's strategy should be a valid probability distribution
        z >= np.min(Q_matrix)  # z should be greater than or equal to the minimum Q-value
    ]
    
    # Expected payoff for each attacker action should be at least z
    for i in range(num_attacker_actions):
        constraints.append(Q_matrix[i] @ q <= z)

    # Form and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract the optimal strategy for the defender
    defender_strategy_cvxpy = q.value


    return {
        "Defender's Optimal Strategy": defender_strategy_cvxpy,
        "Payoff": prob.value
    }

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
    reward=None,
    state=None,
):
    keys = jax.random.split(key, num_rollouts)
    args = [
        (env.game_type, params, policy_net, k, epsilon, gamma, render, for_q_value, reward, state)
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


def rollouts(
    env,
    params,
    policy_net,
    num_rollouts,
    key,
    epsilon,
    gamma,
    render=False,
    for_q_value=False,
    reward=None,
):
    keys = jax.random.split(key, num_rollouts)

    all_states = []
    all_actions = []
    all_action_masks = []
    all_returns = []
    all_masks = []
    all_wins = []


    for k in keys:
    
        args = (env.game_type, params, policy_net, k, epsilon, gamma, render, for_q_value, reward)

    
        states, actions, action_masks, returns, padding_mask, wins = env.single_rollout(args)
        all_states.append(states)
        all_actions.append(actions)
        all_action_masks.append(action_masks)
        all_returns.append(returns)
        all_masks.append(padding_mask)
        all_wins.append(wins)



    return (all_states, all_actions, all_action_masks, all_returns, all_masks, all_wins)
   
    

def get_values(env, params, policy_net, grid_state, num_rollouts, key, epsilon, gamma):
        
        #initial_state = env.set(grid_state[0], grid_state[1], grid_state[2], grid_state[3])
        state = env.reset(key)
        # args = (env.game_type, params, policy_net, key, epsilon, gamma, False, False, 0, None)

        # #print('initial_state values', env.state)
        # states, actions, action_masks, returns, masks, wins = env.single_rollout(
        #     args
        # )
        # print('actions')
        # print(actions)
        # attacker_returns = returns["attacker"][0]
        # v = attacker_returns
        # return v

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
            reward=None,
            state=state
        )
        print('actions')
        #print(actions)
        attacker_returns = [r["attacker"][0] for r in returns]
        v = np.mean(attacker_returns)
        return v

def get_q_values(env, params, policy_net, grid_state, num_rollouts, key, epsilon, gamma):

    # move_list = []
    # _ = env.reset()
    # for d_action in range(env.num_actions):
    #     initial_state = env.reset()
    #     #print('initial_state', initial_state)

    #     #initial_state = env.set(grid_state[0], grid_state[1], grid_state[2], grid_state[3])
    #     #print('initial_state q_values', initial_state)
    #     defender_state, d_reward, _, _ = env.step(
    #         initial_state, 0, "defender", update_env=False
    #     )
    #     defender_state_cp = copy.deepcopy(defender_state)
    #     _ = env.reset()
    #     for a_action in range(env.num_actions):
    #         _ = env.reset()
    #         attacker_state, reward, _, _ = env.step(
    #             defender_state_cp, 3, "attacker", update_env=False
    #         )
    #         _ = env.set_to(defender_state_cp)
    #         attacker_state_cp = copy.deepcopy(attacker_state)
    #         print('state_cp', attacker_state_cp)
    #         args = (env.game_type, params, policy_net, key, epsilon, gamma, False, True, reward, attacker_state_cp)

    #         states, actions, action_masks, returns, masks, wins = env.single_rollout(args)

    attacker_moves = [0,1,2,3]
    defender_moves = [0,1,2,3]
    move_list = []

    #cross product of moves
    moves = itertools.product(defender_moves, attacker_moves)

    for move in moves:
        initial_state = env.reset(key)
        defender_state, d_reward, _, _ = env.step(initial_state, move[0], "defender", update_env=False)
        attacker_state, reward, _, _ = env.step(defender_state, move[1], "attacker", update_env=False)
        args = (env.game_type, params, policy_net, key, epsilon, gamma, False, True, reward, attacker_state)
        #states, actions, action_masks, returns, masks, wins = env.single_rollout(args)
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
                reward=reward,
                state=attacker_state
            )
            
            
        # attacker_returns = returns["attacker"][0]
        # print(attacker_returns)
        # mean_attacker_returns = attacker_returns

        attacker_returns = [r["attacker"][0] for r in returns]
        mean_attacker_returns = np.mean(attacker_returns)
        move_list.append(
            {
                "defender": move[0],
                "attacker": move[1],
                "q_value": mean_attacker_returns,
            }
        )
    print('move_list')
    print(move_list)
    q_values = np.array([move["q_value"] for move in move_list]).reshape(
        env.num_actions, env.num_actions
    )
    print(q_values)

    best_attacker_moves = np.argmax(q_values, axis=1)
    best_defender_move = np.argmin(np.max(q_values, axis=1))
    best_attacker_move = best_attacker_moves[best_defender_move]
    arg_min_max_q = q_values[best_defender_move][best_attacker_move]

    mixed_q = solve_stackelberg_game(move_list)['Payoff']

    return arg_min_max_q, mixed_q

def calc_bellman_error(env, params, policy_net, num_rollouts, key, epsilon, gamma):
    x_a = np.linspace(1, 1, 1)
    y_a = np.linspace(1, 1, 1)
    x_d = np.linspace(8, 8, 1)
    y_d = np.linspace(8, 8, 1)


    xxa, yya, xxd, yyd = np.meshgrid(x_a, y_a, x_d, y_d)
    grid = np.vstack(
        [
            xxa.ravel(),
            yya.ravel(),
            xxd.ravel(),
            yyd.ravel(),
        ]
    ).T

    v_vector = []
    arg_min_max_q_vector = []
    mixed_q_vector = []
    for g in tqdm(grid):
        v = get_values(
            env, params, policy_net, g, num_rollouts, key, epsilon, gamma
        )
        v_vector.append(v)

        arg_min_max_q, mixed_q = get_q_values(
            env, params, policy_net, g, num_rollouts, key, epsilon, gamma
        )
        arg_min_max_q_vector.append(arg_min_max_q)
        mixed_q_vector.append(mixed_q)

    return np.array(arg_min_max_q_vector), mixed_q_vector ,np.array(v_vector)#np.linalg.norm(np.array(q_vector) - np.array(v_vector))


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
                hk.Linear(32),
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
            #arg_min_max_q, mixed_q, v = calc_bellman_error(env, loaded_params, policy_net, num_eval_episodes, jax.random.PRNGKey(episode), epsilon_start, gamma)

            key = jax.random.PRNGKey(episode)
            args = (env.game_type, loaded_params, policy_net, key, epsilon_start, gamma, False, False, 0, None)
                
            states, actions, action_masks, returns, padding_mask, wins = env.single_rollout(args)

            print('Values Rollout')
            print('actions', actions)
            print('returns', returns)

            reward_v = returns['attacker'][0]
            

            print('Q Values Rollout')
            state = env.reset(key)

            state, reward, _, _ = env.step(
                state, 0, "defender", update_env=True
            )
            state, reward, _, _ = env.step(
                state, 3, "attacker", update_env=True
            )

            args = (env.game_type, loaded_params, policy_net, key, epsilon_start, gamma, False, True, reward, None)
            states, actions, action_masks, returns, padding_mask, wins = env.single_rollout(args)
            reward_q = returns['attacker'][0]

            print('actions', actions)
            print('returns', returns)

            print('Bellman Error')
            print(reward_v - reward_q)

            arg_min_max_q, mixed_q, v = calc_bellman_error(env, loaded_params, policy_net, num_eval_episodes, jax.random.PRNGKey(episode), epsilon_start, gamma)
            arg_q_norm = np.linalg.norm(arg_min_max_q)
            mixed_q_norm = np.linalg.norm(mixed_q)
            v_norm = np.linalg.norm(v)
            arg_bellman_error = np.linalg.norm(arg_min_max_q-v)
            mixed_bellman_error = np.linalg.norm(mixed_q-v)
            print('argq:', arg_min_max_q)
            print('mixedq:', mixed_q)
            print('v:', v)
            print('bellman_error:', mixed_bellman_error)

            writer.add_scalars(
                "values",
                {
                    "arg_q_norm": arg_q_norm,
                    "mixed_q_norm": mixed_q_norm,
                    "v_norm": v_norm,
                    "arg bellman": arg_bellman_error,
                    "mixed bellman": mixed_bellman_error,
                },
                episode,)





      




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
    
    game_type = 'stackelberg'
    timestamp = str(datetime.datetime.now())

    env = TwoPlayerGridWorldEnv(
        game_type=game_type,
        grid_size=16,
        init_defender_position=[8,8],
        init_attacker_position=[1,1],
        goal_position=[8,8],
        capture_radius=2
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
    num_parallel = 32 #config['training']['num_parallel'] #mp.cpu_count()
    if num_parallel != config['training']['num_parallel']: 
        ValueError("num_parallel in config file does not match the number of cores on this machine")
    print("cpu_count", num_parallel, "config", config['training']['num_parallel'])

    # Evaluation parameters
    eval_interval = config['eval']['eval_interval']
    num_eval_episodes = config['eval']['num_eval_episodes']

    # Logging
    print(game_type, " starting experiment at :", timestamp)
    writer = SummaryWriter(f"runs_grid2/experiment_{game_type}" + timestamp+"_bellman_error_fix")

    import glob
    import pickle

    # Get a list of all files in the directory
    #files = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_stackelberg/2023-09-19 15:18:57.090737_episode_*_params.pickle')
    files = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_grid_stackelberg/2023-12-28 01:02:09.757833_episode_*_params.pickle')

    files.sort(key=lambda x: int(x.split('_episode_')[1].split('_params')[0]))
    files = files
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