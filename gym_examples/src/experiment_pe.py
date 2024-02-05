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
from envs.two_player_dubins_car_pe import TwoPlayerDubinsCarPEEnv
from matplotlib import cm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import yaml


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
    loaded_params,
):
    # Define loss function
    # Define loss function
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
        print('baseline_values', baseline_values)
        print('returns', returns)
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
    def update_defender(params, value_params, opt_state, observations, actions, returns, mask):
        grads = jax.grad(loss_defender)(params, value_params, observations, actions, returns, mask)
        updates, opt_state = optimizer["defender"].update(
            grads, params=params, state=opt_state
        )
        return optax.apply_updates(params, updates), opt_state, grads

    @jax.jit
    def update_attacker(params, value_params, opt_state, observations, actions, returns, mask):
        grads = jax.grad(loss_attacker)(params, value_params, observations, actions, returns, mask)
        updates, opt_state = optimizer["attacker"].update(
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
        rewards = None,
        state = None
    ):
        keys = jax.random.split(key, num_rollouts)
        args = [
            ("nash", params, policy_net, k, epsilon, gamma, render, for_q_value, rewards, state)
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
            all_returns,
            all_masks,
            all_wins,
        )  # no need to action masks

    def get_values(env, params, policy_net, num_rollouts, key, epsilon, gamma):
        states, actions, returns, masks, wins = parallel_rollouts(
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

    def get_q_values(env, params, policy_net, num_rollouts, key, epsilon, gamma):
        initial_state = env.state #env.reset()
        print('initial_state', initial_state)
        move_list = []
        for d_action in range(env.action_space["defender"].n):
            defender_state, d_reward, _, _ = env.step(
                initial_state, d_action, "defender", update_env=True
            )
            for a_action in range(env.action_space["attacker"].n):
                state, reward, _, _ = env.step(
                    defender_state, a_action, "attacker", update_env=True
                )
                states, actions, returns, masks, wins = parallel_rollouts(
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

        q_values = np.array([move["q_value"] for move in move_list]).reshape(3, 3)
        best_attacker_moves = np.argmax(q_values, axis=1)
        #best_attacker_moves = np.ravel(np.where(q_values[:,0]))
        best_defender_move = np.argmin(np.max(q_values, axis=1))
        best_attacker_move = best_attacker_moves[best_defender_move]
        q = q_values[best_defender_move][best_attacker_move]

        return q

    def calc_bellman_error(env, params, policy_net, num_rollouts, key, epsilon, gamma):
        x_a = np.linspace(-2, 2, 4)
        y_a = np.linspace(2, 2, 1)
        theta_a = np.linspace(0, 0, 1)
        x_d = np.linspace(0, 0, 1)
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
            state = env.set(g[0], g[1], g[2], g[3], g[4], g[5])
            v = get_values(
                env, params, policy_net, num_rollouts, jax.random.PRNGKey(42), 0.01, 0.95
            )
            v_vector.append(v)

            state = env.set(g[0], g[1], g[2], g[3], g[4], g[5])
            q = get_q_values(
                env, params, policy_net, num_rollouts, jax.random.PRNGKey(42), 0.01, 0.95
            )
            q_vector.append(q)


        return np.linalg.norm(np.array(q_vector) - np.array(v_vector))

    def save_params(episode_num, params, value_params, game_type, timestamp):
        with open(
            f"data/pe_{game_type}/{timestamp}_episode_{episode_num}_params.pickle", "wb"
        ) as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(
            f"data/pe_{game_type}/{timestamp}_episode_{episode_num}_value_params.pickle", "wb"
        ) as handle:
            pickle.dump(value_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    
    ############################
    # Initialize Haiku policy network
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    value_net = hk.without_apply_rng(hk.transform(value_network))
    initial_state = env.reset()
    initial_state_nn = env.encode_helper(initial_state)

    if loaded_params is None:
        params = {
            player: policy_net.init(jax.random.PRNGKey(42), initial_state_nn)
            for player in env.players
        }
        value_params = value_net.init(jax.random.PRNGKey(42), initial_state_nn)
    else:
        params = loaded_params
        print("loaded params")

    # Define the optimizer
    agent_optimizer = optax.chain(
        optax.radam(learning_rate=learning_rate, b1=0.9, b2=0.9)
    )
    optimizer = {player: agent_optimizer for player in env.players}
    opt_state = {
        player: optimizer[player].init(params[player]) for player in env.players
    }
    value_optimizer = agent_optimizer
    value_opt_state = value_optimizer.init(value_net.init(jax.random.PRNGKey(42), initial_state_nn))

    

    episode_losses = []  # Add this line to store the losses
    all_wins, all_traj_lengths = [], []
    batch_states = {player: [] for player in env.players}
    batch_actions = {player: [] for player in env.players}
    batch_returns = {player: [] for player in env.players}
    batch_wins = {player: [] for player in env.players}
    batch_masks = {player: [] for player in env.players}
    traj_length = []
    epsilon = epsilon_start

    ##################################
    # Start the main loop over episodes
    for episode in range(num_episodes):
        print(f"Episode {episode} started...")
        key = jax.random.PRNGKey(episode)
        valid = True
        render = False

        env.reset(key)
        states, actions, returns, masks, wins = parallel_rollouts(
            env, params, policy_net, num_parallel, key, epsilon, gamma
        )
        for (
            rollout_states,
            rollout_actions,
            rollout_returns,
            rollout_masks,
            rollout_wins,
        ) in zip(states, actions, returns, masks, wins):
            for player in env.players:
                batch_states[player].append(rollout_states[player])
                batch_actions[player].append(rollout_actions[player])
                batch_returns[player].append(rollout_returns[player])
                batch_wins[player].append(rollout_wins[player])
                batch_masks[player].append(rollout_masks[player])
        # print('batch_states')
        # pprint(batch_states['defender'])
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        # Initialize a Counter
        wins_ctr = Counter()
        # Add up the wins from each dictionary
        for win_dict in wins:
            wins_ctr += Counter(win_dict)

        writer.add_scalar('average returns', np.array(np.mean(batch_returns['attacker'])), episode)


        # print(f"Episode {episode} finished", 'returns:', np.mean(batch_returns[player]))
        # print('num wins', wins_ctr)
        # print('average_length', np.mean(traj_length))

        # writer.add_scalars('returns', {'defender': np.array(np.mean(batch_returns['defender'])),
        #                        'attacker': np.array(np.mean(batch_returns['attacker']))},
        #            episode)

        writer.add_scalars(
            "num_wins",
            {
                "defender": np.array(wins_ctr["defender"]),
                "attacker": np.array(wins_ctr["attacker"]),
                "draw": np.array(wins_ctr["draw"]),
            },
            episode,
        )

        if episode % eval_interval == 0:
            _, _, _, _, _, _ = env.single_rollout_nash(
                [
                    "nash",
                    params,
                    policy_net,
                    jax.random.PRNGKey(episode),
                    epsilon,
                    gamma,
                    True,
                    False,
                    None,
                    None
                ]
            )
            env.make_gif(f"gifs/persuit_evasion/{timestamp}_{episode}.gif")
            save_params(episode, params, value_params, "nash", timestamp)

            # bellman_error = calc_bellman_error(env, params, policy_net, num_eval_episodes, jax.random.PRNGKey(episode), epsilon, gamma)
            # print('bellman_error', bellman_error)
            # writer.add_scalar('bellman_error', bellman_error, episode)

        if (episode + 1) % batch_multiple == 0 and valid:
            for player in env.players:
                print("training")

                if player == "defender":
                    params[player], opt_state[player], defender_grads = update_defender(
                        params[player],
                        value_params,
                        opt_state[player],
                        np.array(batch_states[player]),
                        np.array(batch_actions[player]),
                        np.array(batch_returns[player]),
                        np.array(batch_masks[player]),
                    )
                elif player == "attacker":
                    params[player], opt_state[player], attacker_grads = update_attacker(
                        params[player],
                        value_params,
                        opt_state[player],
                        np.array(batch_states[player]),
                        np.array(batch_actions[player]),
                        np.array(batch_returns[player]),
                        np.array(batch_masks[player]),
                    )

            value_params, value_opt_state, = update_value_network(
                value_params, value_opt_state, 
                np.array(batch_states['attacker']), 
                np.array(batch_returns['attacker']), 
                np.array(batch_masks['attacker'])
            )

            value_norm = optax.global_norm(value_params)
            current_value_loss = value_loss(value_params, np.array(batch_states['attacker']), np.array(batch_returns['attacker']), np.array(batch_masks['attacker']))
            print('current_value_loss', current_value_loss)
            writer.add_scalar('value_loss', np.array(current_value_loss), episode)

            defender_norm = optax.global_norm(params["defender"])
            attacker_norm = optax.global_norm(params["attacker"])
            defender_grad_norm = optax.global_norm(defender_grads)
            attacker_grad_norm = optax.global_norm(attacker_grads)
            writer.add_scalars(
                "norms",
                {
                    "defender": np.array(defender_norm),
                    "attacker": np.array(attacker_norm),
                },
                episode,
            )
            writer.add_scalars(
                "grad_norms",
                {
                    "defender": np.array(defender_grad_norm),
                    "attacker": np.array(attacker_grad_norm),
                },
                episode,
            )

            all_wins.append(wins)
            #all_traj_lengths.append(np.mean(traj_length))
            batch_states = {player: [] for player in env.players}
            batch_actions = {player: [] for player in env.players}
            batch_returns = {player: [] for player in env.players}
            batch_wins = {player: [] for player in env.players}
            batch_masks = {player: [] for player in env.players}
            wins = 0
            traj_length = []

    return params


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
    loaded_params,
):
    # Define loss function
    # Define loss function
    @jax.jit
    def loss_attacker(
        params, value_params, observations, actions, action_masks, returns, padding_mask
    ):
        action_probabilities = policy_net.apply(params, observations, action_masks)
        #action_probabilities = jax.nn.softmax(action_probabilities)
        log_probs = jnp.log(jnp.take_along_axis(
                action_probabilities + 10e-6, actions[..., None], axis=-1
            )
        )

        # Get baseline values
        baseline_values = value_net.apply(value_params, observations).squeeze(-1)
        print('baseline_values', baseline_values)
        print('returns', returns)
        advantage = returns - baseline_values
            
        log_probs = log_probs.reshape(returns.shape)
        masked_loss = padding_mask * (-log_probs * jax.lax.stop_gradient(advantage))
        return jnp.sum(masked_loss) / jnp.sum(padding_mask)

    @jax.jit
    def loss_defender(
        params, value_params, observations, actions, action_masks, returns, padding_mask
    ):
        action_probabilities = policy_net.apply(params, observations, action_masks)
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
        params, value_params, opt_state, observations, actions, action_masks, returns, padding_mask
    ):
        grads = jax.grad(loss_defender)(
            params, value_params, observations, actions, action_masks, returns, padding_mask
        )
        updates, opt_state = optimizer["defender"].update(
            grads, params=params, state=opt_state
        )
        return optax.apply_updates(params, updates), opt_state, grads

    @jax.jit
    def update_attacker(
        params, value_params, opt_state, observations, actions, action_masks, returns, padding_mask
    ):
        grads = jax.grad(loss_attacker)(
            params, value_params, observations, actions, action_masks, returns, padding_mask
        )
        updates, opt_state = optimizer["attacker"].update(
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
        state = None
    ):
        keys = jax.random.split(key, num_rollouts)
        args = [
            ("stackelberg", params, policy_net, k, epsilon, gamma, render, for_q_value, reward, state)
            for k in keys
        ]
        with ProcessPool() as pool:
            results = pool.map(env.single_rollout_stackelberg, args)
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

    def get_values(env, params, policy_net, num_rollouts, key, epsilon, gamma):
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

    def get_q_values(env, params, policy_net, num_rollouts, key, epsilon, gamma):
        initial_state = env.reset()
        move_list = []
        for d_action in range(env.action_space["defender"].n):
            defender_state, d_reward, _, _ = env.step(
                initial_state, d_action, "defender", update_env=True
            )
            for a_action in range(env.action_space["attacker"].n):
                state, reward, _, _ = env.step(
                    defender_state, a_action, "attacker", update_env=True
                )
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

        q_values = np.array([move["q_value"] for move in move_list]).reshape(
            env.num_actions, env.num_actions
        )
        best_attacker_moves = np.argmax(q_values, axis=1)
        best_defender_move = np.argmin(np.max(q_values, axis=1))
        best_attacker_move = best_attacker_moves[best_defender_move]
        q = q_values[best_defender_move][best_attacker_move]
        return q

    def calc_bellman_error(env, params, policy_net, num_rollouts, key, epsilon, gamma):
        x_a = np.linspace(2, 2, 1)
        y_a = np.linspace(2, 2, 1)
        theta_a = np.linspace(0, 0, 1)
        x_d = np.linspace(0, 0, 1)
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
            state = env.set(g[0], g[1], g[2], g[3], g[4], g[5])
            v = get_values(
                env, params, policy_net, num_rollouts, jax.random.PRNGKey(42), 0.01, 0.95
            )
            v_vector.append(v)

            state = env.set(g[0], g[1], g[2], g[3], g[4], g[5])
            q = get_q_values(
                env, params, policy_net, num_rollouts, jax.random.PRNGKey(42), 0.01, 0.95
            )
            q_vector.append(q)

        return np.linalg.norm(np.array(q_vector) - np.array(v_vector))

    def save_params(episode_num, params, value_params, game_type, timestamp):
        with open(
            f"data/experiment_{game_type}/{timestamp}_episode_{episode_num}_params.pickle", "wb"
        ) as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(
            f"data/experiment_{game_type}/{timestamp}_episode_{episode_num}_value_params.pickle", "wb"
        ) as handle:
            pickle.dump(value_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    ############################
    # Initialize Haiku policy network
    policy_net = hk.without_apply_rng(hk.transform(policy_network))
    value_net = hk.without_apply_rng(hk.transform(value_network))

    initial_state = env.reset()
    initial_state_nn = env.encode_helper(initial_state)

    if loaded_params is None:
        params = {
            player: policy_net.init(
                jax.random.PRNGKey(42),
                initial_state_nn,
                env.get_legal_actions_mask(initial_state, player),
            )
            for player in env.players
        }

        value_params = value_net.init(jax.random.PRNGKey(42), initial_state_nn)


    else:
        params = loaded_params
        print("loaded params")

    # Define the optimizer
    agent_optimizer = optax.chain(
        optax.radam(learning_rate=learning_rate, b1=.9, b2=.9),
        #optax.optimistic_gradient_descent(learning_rate=learning_rate),
    )
    optimizer = {player: agent_optimizer for player in env.players}
    opt_state = {
        player: optimizer[player].init(params[player]) for player in env.players
    }

    value_optimizer = agent_optimizer
    value_opt_state = value_optimizer.init(value_net.init(jax.random.PRNGKey(42), initial_state_nn))

    episode_losses = []  # Add this line to store the losses
    all_wins, all_traj_lengths = [], []
    batch_states = {player: [] for player in env.players}
    batch_actions = {player: [] for player in env.players}
    batch_action_masks = {player: [] for player in env.players}
    batch_returns = {player: [] for player in env.players}
    batch_wins = {player: [] for player in env.players}
    batch_masks = {player: [] for player in env.players}
    traj_length = []
    epsilon = epsilon_start
    training_player = 'attacker'  # start with the first player (the defender)
    training_counter = 1
    defender_norm = [0]
    attacker_norm = [0]
    defender_grad_norm = [0]
    attacker_grad_norm = [0]


    ##################################
    # Start the main loop over episodes
    for episode in range(num_episodes):
        print(f"Episode {episode} started...")
        key = jax.random.PRNGKey(episode)
        valid = True
        render = False

        env.reset(key)
        states, actions, action_masks, returns, masks, wins = parallel_rollouts(
            env, params, policy_net, num_parallel, key, epsilon, gamma
        )
        for (
            rollout_states,
            rollout_actions,
            rollout_action_masks,
            rollout_returns,
            rollout_masks,
            rollout_wins,
        ) in zip(states, actions, action_masks, returns, masks, wins):
            for player in env.players:
                batch_states[player].append(rollout_states[player])
                batch_actions[player].append(rollout_actions[player])
                batch_action_masks[player].append(rollout_action_masks[player])
                batch_returns[player].append(rollout_returns[player])
                batch_wins[player].append(rollout_wins[player])
                batch_masks[player].append(rollout_masks[player])

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        # Initialize a Counter
        wins_ctr = Counter()
        # Add up the wins from each dictionary
        for win_dict in wins:
            wins_ctr += Counter(win_dict)

        writer.add_scalars(
            "num_wins",
            {
                "defender": np.array(wins_ctr["defender"]),
                "attacker": np.array(wins_ctr["attacker"]),
                "draw": np.array(wins_ctr["draw"]),
            },
            episode,
        )

        writer.add_scalar('average returns', np.array(np.mean(batch_returns['attacker'])), episode)

        if episode % eval_interval == 0:
            (
                states,
                actions,
                action_masks,
                returns,
                padding_mask,
                wins,
            ) = env.single_rollout(
                [
                    'stackelberg',
                    params,
                    policy_net,
                    jax.random.PRNGKey(episode),
                    epsilon,
                    gamma,
                    True,
                    False,
                    None,
                    None
                ]
            )
            env.make_gif(f"gifs/persuit_evasion/{timestamp}_{episode}.gif")
            save_params(episode, params,value_params, "stackelberg", timestamp)

            # bellman_error = calc_bellman_error(env, params, policy_net, num_eval_episodes, jax.random.PRNGKey(episode), epsilon, gamma)
            # print('bellman_error', bellman_error)
            # writer.add_scalar('bellman_error', bellman_error, episode)
            
            # if training_counter % 4 == 0:
            #     training_player = 'attacker'
            # else:
            #     training_player = 'defender'
            # training_counter += 1
            # training_player = env.players[
            #     (env.players.index(training_player) + 1) % len(env.players)
            # ]  # switch trining player
        #     training_player = 'defender' 
        # else:
        #     training_player = 'attacker'

        #this is the original
        if episode % 4 == 0:
            training_player = 'defender'
        else:
            training_player = 'attacker'

        #alternate players every 4th episode
        # if episode % 4 == 0:
        #     training_player = env.players[
        #          (env.players.index(training_player) + 1) % len(env.players)
        #     ]  # switch trining player




        if (episode + 1) % batch_multiple == 0 and valid:
            print("training", training_player)
            if training_player == "defender":
                (
                    params["defender"],
                    opt_state["defender"],
                    defender_grads,
                ) = update_defender(
                    params["defender"],
                    value_params,
                    opt_state["defender"],
                    np.array(batch_states["defender"]),
                    np.array(batch_actions["defender"]),
                    np.array(batch_action_masks["defender"]),
                    np.array(batch_returns["defender"]),
                    np.array(batch_masks["defender"]),
                )
                defender_norm = optax.global_norm(params["defender"])
                defender_grad_norm = optax.global_norm(defender_grads)

            elif training_player == "attacker":
                (
                    params["attacker"],
                    opt_state["attacker"],
                    attacker_grads,
                ) = update_attacker(
                    params["attacker"],
                    value_params,
                    opt_state["attacker"],
                    np.array(batch_states["attacker"]),
                    np.array(batch_actions["attacker"]),
                    np.array(batch_action_masks["attacker"]),
                    np.array(batch_returns["attacker"]),
                    np.array(batch_masks["attacker"]),
                )

                attacker_norm = optax.global_norm(params["attacker"])
                attacker_grad_norm = optax.global_norm(attacker_grads)
            
            value_params, value_opt_state, = update_value_network(
                value_params, value_opt_state, 
                np.array(batch_states['attacker']), 
                np.array(batch_returns['attacker']), 
                np.array(batch_masks['attacker'])
            )

            value_norm = optax.global_norm(value_params)
            current_value_loss = value_loss(value_params, np.array(batch_states['attacker']), np.array(batch_returns['attacker']), np.array(batch_masks['attacker']))
            print('current_value_loss', current_value_loss)
            writer.add_scalar('value_loss', np.array(current_value_loss), episode)


            writer.add_scalars(
                "norms",
                {
                    "defender": np.array(defender_norm),
                    "attacker": np.array(attacker_norm),
                },
                episode,
            )
            writer.add_scalars(
                "grad_norms",
                {
                    "defender": np.array(defender_grad_norm),
                    "attacker": np.array(attacker_grad_norm),
                },
                episode,
            )

            all_wins.append(wins)
            #all_traj_lengths.append(np.mean(traj_length))
            batch_states = {player: [] for player in env.players}
            batch_actions = {player: [] for player in env.players}
            batch_action_masks = {player: [] for player in env.players}
            batch_returns = {player: [] for player in env.players}
            batch_wins = {player: [] for player in env.players}
            batch_masks = {player: [] for player in env.players}
            wins = 0
            traj_length = []

    return params



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
    config = load_config("configs/config_pe.yml")
    print_config(config)
    
    game_type = config['game']['type']
    timestamp = str(datetime.datetime.now())

    env = TwoPlayerDubinsCarPEEnv(
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
    writer = SummaryWriter(f"runs_10_baseline/experiment_{game_type}" + timestamp+"persuit-evasion")
    #Load data (deserialize)
    # with open('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_stackelberg/2023-09-15 10:22:44.481035_episode_12224_params.pickle', 'rb') as handle:
    #     loaded_params = pickle.load(handle)

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
            loaded_params=None,
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
            loaded_params=None,
        )


#don't have shedule for epsilon decay, just fix it.  upghrade learning rate

# measure the right error, and make sure it goes down. 
# bayseian persuasion, principle agent 