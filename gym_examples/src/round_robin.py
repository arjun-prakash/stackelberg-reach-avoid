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

import cvxpy as cp


def play_match(
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
    params_list_a,
    params_list_d,
    player_types,
    salt,
):
    # Define loss function
    # Define loss function

    

    def policy_network_stackelberg(observation, legal_moves):
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
                hk.Linear(env.num_actions),
                jax.nn.softmax,
            ]
        )

        return net(observation)

    ############################
    # Initialize Haiku policy network
    policy_net_stackelberg = hk.without_apply_rng(hk.transform(policy_network_stackelberg))
    policy_net_nash = hk.without_apply_rng(hk.transform(policy_network_nash))


    initial_state = env.reset()

    for file in params_list_a:
        with open(file, 'rb') as handle:
            loaded_params_a= pickle.load(handle)
            episode = int(file.split('_episode_')[1].split('_params')[0])
            key = jax.random.PRNGKey(episode)

    for file in params_list_d:
        with open(file, 'rb') as handle:
            loaded_params_d = pickle.load(handle)
            episode = int(file.split('_episode_')[1].split('_params')[0])
            key = jax.random.PRNGKey(episode)


    win_ctr= Counter()
    win_lengths = []
    other_lengths = []

    for episode in range(50):
        key = jax.random.PRNGKey((episode+1)*salt)
        _ = env.reset(key)
        (
            states,
            actions,
            action_masks,
            returns,
            padding_mask,
            wins,
            step
        ) = env.single_rollout_round_robin(
            [
                '',
                loaded_params_a,
                loaded_params_d,
                policy_net_stackelberg,
                policy_net_nash,
                player_types,
                key,
                epsilon_start,
                gamma,
                False,
                False,
                None,
            ]
        )
        #env.make_gif(f"gifs/round_robin/{timestamp}_{(episode+1)*salt}_scloser.gif")
        #print("Episode", episode, "Winner", wins)
        #print('length', step)
        if wins['attacker'] == 1:
            win_lengths.append(step)
        else:
            other_lengths.append(step)
        win_ctr += Counter(wins)

    print(win_ctr)
    print('average win length', np.mean(win_lengths))
    print('std win length', np.std(win_lengths))
    print('average other length', np.mean(other_lengths))
    print('std pther length', np.std(other_lengths))

      




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
    
    game_type = 'stackelberg'#config['game']['type']
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
    #writer = SummaryWriter(f"runs5/experiment_{game_type}" + timestamp+"_bellman_error_fix")

    import glob
    import pickle

    # Get a list of all files in the directory
    files_s = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_stackelberg/2024-01-06 08:30:16.864426_episode_*_params.pickle') #best stackelberg baseline for final (bounded)
    #files_s = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_stackelberg/2024-01-22 17:27:34.350206_episode_*_params.pickle') #stack no boundary

    files_s = files_s[::2]

    files_s.sort(key=lambda x: int(x.split('_episode_')[1].split('_params')[0]))
    files_s= [files_s[20]]
    print(files_s)

    files_n = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_nash/2024-01-13 04:14:42.638173_episode_*_params.pickle') #new nash baseline for final (bounded)
    #files_n = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/experiment_nash/2024-01-21 12:12:21.400600_episode_*_params.pickle') #unbounded
    files_n= files_n[::2]
    files_n.sort(key=lambda x: int(x.split('_episode_')[1].split('_params')[0]))
    files_n= [files_n[20]]
    print(files_n)

    files_p = glob.glob('/users/apraka15/arjun/gym-examples/gym_examples/src/data/pe_nash/2024-02-05 17:27:26.792384_episode_*_params.pickle') #nash pe
    files_p = files_p[::2]
    files_p.sort(key=lambda x: int(x.split('_episode_')[1].split('_params')[0]))
    files_p= [files_p[20]]


    # Now params_list contains the parameters from all episodes



    trained_params = play_match(
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
        writer=None,
        timestamp=timestamp,
        params_list_a=files_s,
        params_list_d=files_s,
        player_types={'attacker': 'stackelberg', 'defender': 'stackelberg'},
        salt=1,
    )


#don't have shedule for epsilon decay, just fix it.  upghrade learning rate

# measure the right error, and make sure it goes down. 
# bayseian persuasion, principle agent 