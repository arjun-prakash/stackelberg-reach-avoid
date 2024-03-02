

import itertools
import gym
from gym import spaces
import numpy as np
import jax
import copy
import jax.numpy as jnp
import haiku as hk
import optax
import jax.numpy as jnp
from jax import jit, vmap


from envs.dubins_car import DubinsCarEnv

class TwoPlayerDubinsCarEnv(DubinsCarEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, game_type, num_actions, size, reward, max_steps, init_defender_position, init_attacker_position, capture_radius, goal_position, goal_radius, timestep, v_max, omega_max):
        super().__init__()

        self.game_type = game_type
        self.players = ['defender', 'attacker']
        self.num_actions = num_actions #3
        self.action_space = {'attacker':spaces.Discrete(self.num_actions), 'defender':spaces.Discrete(self.num_actions)}

        self.size = 3#4
        self.reward = reward #1
        self.max_steps = max_steps# 50

        self.observation_space= {'attacker':spaces.Box(low=np.array([-self.size, -self.size, 0]), high=np.array([self.size,self.size , 2*jnp.pi]), dtype=jnp.float32), 
                                'defender':spaces.Box(low=np.array([-self.size, -self.size, 0]), high=np.array([self.size, self.size, 2*jnp.pi]), dtype=jnp.float32)}



        self.init_defender_position = init_defender_position #jnp.array([0,0,0])
        self.init_attacker_position = init_attacker_position #jnp.array([2,2,0])
        self.state = {'attacker': jnp.array([self.init_attacker_position]), 'defender':jnp.array(self.init_defender_position)}
        self.capture_radius = capture_radius #0.5 # radius of the obstacle

        self.goal_position = jnp.array(goal_position) # position of the goal
        self.goal_radius = goal_radius # minimum distance to goal to consider the task as done

        self.timestep = timestep # timestep in seconds
        self.v_max = v_max # maximum speed
        self.omega_max = omega_max * jnp.pi/180  # maximum angular velocity (radians)
        self.images = []
        self.positions = {'attacker': [], 'defender': []}
        


    def _reset(self, key):
        # Use JAX random to set initial positions if needed
        key, subkey1, subkey2 = jax.random.split(key, 3)
        down = 3 * np.pi / 2
        up = np.pi / 2

        # Example of setting random positions; adapt as necessary for your environment
        defender_x = jax.random.uniform(subkey1, minval=-1, maxval=1)
        defender_y = jax.random.uniform(subkey1, minval=-2, maxval=-2)
        defender_theta = jax.random.uniform(subkey1, minval=up, maxval=up)

        attacker_x = jax.random.uniform(subkey2, minval=-self.size, maxval=self.size)
        attacker_y = jax.random.uniform(subkey2, minval=2, maxval=self.size)
        attacker_theta = jax.random.uniform(subkey2, minval=down, maxval=down)

        # Set initial state using provided positions or randomized ones
        state = {
            'defender': jnp.array([defender_x, defender_y, defender_theta]),
            'attacker': jnp.array([attacker_x, attacker_y, attacker_theta])
        }

        return state, key

    
    def set(self, ax, ay, atheta, dx=0., dy=0., dtheta=0.):
        """
        Reset the environment and return the initial state
        """
        self.state['attacker'] = jnp.array([ax, ay, atheta], dtype=self.observation_space['attacker'].dtype)
        self.state['defender'] = jnp.array([dx, dy, dtheta], dtype=self.observation_space['defender'].dtype)

        self.goal_position = self.goal_position
        return self.state
    

    def _check_boundaries_and_update_theta(self, state, player):
        # Extract player's current position and theta
        x, y, theta = state[player]
        
        # Check x-boundary
        out_of_x_bounds = jnp.logical_or(x <= self.observation_space[player].low[0],
                                        x >= self.observation_space[player].high[0])
        
        # Check y-boundary
        out_of_y_bounds = jnp.logical_or(y <= self.observation_space[player].low[1],
                                        y >= self.observation_space[player].high[1])

        # Check if out of bounds in either x or y direction
        out_of_bounds = jnp.logical_or(out_of_x_bounds, out_of_y_bounds)

        # Update theta if out of bounds
        new_theta = jnp.where(out_of_bounds, (jnp.pi + theta) % (2 * jnp.pi), theta)

        return new_theta

    def _update_state_with_boundaries(self, state, action, player):
        # Your existing state update logic here...
        # For demonstration, assuming new_state is computed as before
        new_state = self._update_state(state, action, player)  # assuming this is your existing function
        
        # Update theta based on boundary conditions
        new_theta = self._check_boundaries_and_update_theta(new_state, player)
        
        # Construct the new player state with updated theta
        new_player_state = jnp.array([new_state[player][0], new_state[player][1], new_theta])
        
        # Update the state dictionary immutably with the new player state
        updated_state = {**new_state, player: new_player_state}
        
        return updated_state




    def _update_state(self, state, action, player):
        # Map actions to omega values
        # omega = jnp.where(action == 0, -self.omega_max, 
        #       jnp.where(action == 2, self.omega_max, 0))

        omega = action[0] * -self.omega_max + action[2] *self.omega_max

        #print('update',state)
        # Compute the new state
        new_theta = state[player][2] + omega * self.timestep
        
        #change v_max depending on player
        v_max = jax.lax.cond(player == 'attacker', lambda _: self.v_max, lambda _: self.v_max, None)

        new_x = state[player][0] + v_max * jnp.cos(new_theta) * self.timestep
        new_y = state[player][1] + v_max * jnp.sin(new_theta) * self.timestep


        # Update the state immutably
        new_player_state = jnp.array([new_x, new_y, new_theta % (2 * jnp.pi)])

        # Update the state dictionary immutably
        new_state = {**state, player: new_player_state}
        return new_state

    #this is real
    def _get_reward_done_nash(self, state, player):
        # Calculate the distance of the attacker from the goal
        dist_goal = jnp.linalg.norm(state['attacker'][:2] - self.goal_position)

        # Check if the attacker has reached the goal

        # Check if the attacker is caught by the defender
        done_win = dist_goal < self.goal_radius 
        done_loss = jnp.linalg.norm(state['attacker'][:2] - state['defender'][:2]) < self.capture_radius 


        # Assign rewards based on the conditions
        reward = jnp.where(done_win, 200, -((dist_goal - self.goal_radius)**2))
        reward = jnp.where(done_loss, -200, reward)


        # Determine if the episode is done (either win or loss)
        done = jnp.logical_or(done_win, done_loss)

        return reward, done

    def _get_reward_done_stack(self, state, player):
        # Calculate the distance of the attacker from the goal
        dist_goal = jnp.linalg.norm(state['attacker'][:2] - self.goal_position)

        # Check if the attacker has reached the goal

        # Check if the attacker is caught by the defender
        done_win = dist_goal < self.goal_radius 


        # Assign rewards based on the conditions
        reward = jnp.where(done_win, 200, -((dist_goal - self.goal_radius)**2))


        # Determine if the episode is done (either win or loss)

        return reward, done_win

    def _get_reward_done_pe(self, state, player):
        # persuit evasion
        dist_capture = jnp.linalg.norm(state['attacker'][:2] - state['defender'][:2])

        # Check if the attacker has reached the goal

        # Check if the attacker is caught by the defender
        done = jnp.linalg.norm(state['attacker'][:2] - state['defender'][:2]) < self.capture_radius 




        # Determine if the episode is done (either win or loss)
      

        return dist_capture, done


    def _reset_if_done(self, key, env_state, done):
        return jax.lax.cond(
            done,
            lambda _: self._reset(key),  # You need to implement this _reset method
            lambda _: env_state,
            None,
        )
    
    def step_nash(self, env_state, action, player):
        state = env_state
        new_state = self._update_state_with_boundaries(state, action, player)
        reward, done = self._get_reward_done_nash(new_state, player)
        #new_state = self._reset_if_done(key, new_state, done)
        nn_state = self.encode_helper(new_state)
        return new_state, nn_state, reward, done
    
    def step_stack(self, env_state, action, player):
        state = env_state
        new_state = self._update_state_with_boundaries(state, action, player)
        reward, done = self._get_reward_done_stack(new_state, player)
        #new_state = self._reset_if_done(key, new_state, done)
        nn_state = self.encode_helper(new_state)
        return new_state, nn_state, reward, done
    
    
    def step_pe(self, env_state, action, player):
        state = env_state
        new_state = self._update_state(state, action, player)
        reward, done = self._get_reward_done_pe(new_state, player)
        #new_state = self._reset_if_done(key, new_state, done)
        nn_state = self.encode_helper(new_state)
        return new_state, nn_state, reward, done
    
    

    def reset(self, key):
        # Implement the reset logic
        env_state = self._reset(key)  # You need to implement this _reset method
        new_state = env_state[0]
        nn_state = self.encode_helper(new_state)
        return env_state, new_state, nn_state




    
    def get_legal_actions_mask2(self, state):
            # Define a helper function to check if the 'attacker' gets captured by the 'defender'
        def apply_actions(state, actions):
            action_defender, action_attacker = actions
            next_state, nn_state, _, _ = self.step_nash(state, action_attacker, 'attacker') #use nash to check the legal actions
            init_state, init_nn_state, reward, done = self.step_nash(next_state, action_defender, 'defender')
            #check reward is -200 and done is true
            is_legal = jax.lax.cond(
                jnp.logical_and(reward == -200, done == True),
                lambda _: 0,
                lambda _: 1,
                None,
            )
            
            return is_legal
        # JIT compile the helper function for efficiency

        defender_actions = jnp.array([0,1,2])
        attacker_actions = jnp.array([0,1,2])
        action_pairs = jnp.array(list(itertools.product(defender_actions, attacker_actions)))

        # Vectorize the apply_actions function to handle all action pairs in parallel
        batch_apply_actions = jax.vmap(apply_actions, in_axes=(None, 0))

        # Apply all action pairs to the initial state

        is_legal = batch_apply_actions(state, action_pairs)
        is_legal = is_legal.reshape(len(defender_actions), len(attacker_actions))

        #check all cols are 1
        mask = jnp.all(is_legal, axis=0)

        
        return mask
    
    def get_legal_actions_mask1(self, state):
            # Define a helper function to check if the 'attacker' gets captured by the 'defender'
        def apply_actions(state, action):
            next_state, nn_state, reward, done = self.step_nash(state, action, 'attacker')
            #check reward is -200 and done is true
            is_legal = jax.lax.cond(
                jnp.logical_and(reward == -200, done == True),
                lambda _: 0,
                lambda _: 1,
                None,
            )
            
            return is_legal
        # JIT compile the helper function for efficiency

        attacker_actions = jnp.array([[1,0,0],[0,1,0],[0,0,1]])

        # Vectorize the apply_actions function to handle all action pairs in parallel
        batch_apply_actions = jax.vmap(apply_actions, in_axes=(None, 0))

        # Apply all action pairs to the initial state

        is_legal = batch_apply_actions(state, attacker_actions)

        
        
        return is_legal
    









    def state_for_env(self,nn_state):
        """convert the state from the neural network np.array representation to the environment representation dict"""

        env_state = {'attacker':np.array([nn_state[0], nn_state[1], np.arctan2(nn_state[3], nn_state[2])]), 'defender':np.array([nn_state[4], nn_state[5], np.arctan2(nn_state[7], nn_state[6])])}
        return env_state

    def state_for_nn(self,env_state):
        """convert the state from the environment (dict) representation to the neural network representation np.array"""
        nn_state = np.array([env_state['attacker'][0], env_state['attacker'][1], np.cos(env_state['attacker'][2]), np.sin(env_state['attacker'][2]), env_state['defender'][0], env_state['defender'][1], np.cos(env_state['defender'][2]), np.sin(env_state['defender'][2])])
        return nn_state
    



    

   

            


    def render(self, states, dones, mode='human', close=False):
        """
        Render the environment for human viewing
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import io
        import imageio
        matplotlib.use('Agg')



        fig = plt.figure()
        plt.clf()

        #xlim max of size or position
        
        i = 0
        for state in states:

            if dones[i] == True:
                break
            i+=1

            xlim = max(self.size, max([abs(state['attacker'][0]) for state in states]), max([abs(state['defender'][0]) for state in states]))
            ylim = max(self.size, max([abs(state['attacker'][1]) for state in states]), max([abs(state['defender'][1]) for state in states]))

            plt.xlim([-xlim, xlim])
            plt.ylim([-ylim, ylim])

            self.positions['attacker'].append((state['attacker'][:2].copy(), state['attacker'][2]))
            self.positions['defender'].append((state['defender'][:2].copy(), state['defender'][2]))




            for player, pos_orients in self.positions.items():
                color = 'b' if player == 'attacker' else 'r'
                for pos, orient in pos_orients:
                    dx = 0.1 * np.cos(orient)
                    dy = 0.1 * np.sin(orient)
                    plt.arrow(*pos, dx, dy, color=color, head_width=0.1, head_length=0.1)
        

            attacker = plt.Circle((state['attacker'][0], state['attacker'][1]), 0.1, color='b', fill=True)
            defender = plt.Circle((state['defender'][0], state['defender'][1]), self.capture_radius, color='r', fill=True)
            defender_capture_radius = plt.Circle((state['defender'][0], state['defender'][1]), self.capture_radius+self.v_max, color='r', fill=False, linestyle='--')



            # draw goal
            goal = plt.Circle((self.goal_position[0], self.goal_position[1]), self.goal_radius, color='g', fill=False)

            # draw obstacle
            plt.gca().add_artist(attacker)
            plt.gca().add_artist(defender)
            plt.gca().add_artist(goal)
            plt.gca().add_artist(defender_capture_radius)

            

            #print('saving')
            # save current figure to images list
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.images.append(np.array(imageio.v2.imread(buf)))
            plt.close()

    def make_gif(self, file_name='two_player_animation_pg.gif'):
        import imageio

        end_state = self.images[-1]
        imageio.mimsave(file_name, self.images, fps=30)
        self.images = []
        self.positions = {'attacker': [], 'defender': []}

        return end_state




    def norm_state_for_env(self, nn_state):
        min_value = -4
        max_value = 4

        env_state = {
            'attacker': np.array([
                nn_state[0] * (max_value - min_value) + min_value,
                nn_state[1] * (max_value - min_value) + min_value,
                np.arctan2(nn_state[3], nn_state[2])
            ]),
            'defender': np.array([
                nn_state[4] * (max_value - min_value) + min_value,
                nn_state[5] * (max_value - min_value) + min_value,
                np.arctan2(nn_state[7], nn_state[6])
            ])
        }
        return env_state


    def norm_state_for_nn(self, env_state):
        min_value = -4
        max_value = 4

        nn_state = np.array([
            (env_state['attacker'][0] - min_value) / (max_value - min_value),
            (env_state['attacker'][1] - min_value) / (max_value - min_value),
            np.cos(env_state['attacker'][2]),
            np.sin(env_state['attacker'][2]),
            (env_state['defender'][0] - min_value) / (max_value - min_value),
            (env_state['defender'][1] - min_value) / (max_value - min_value),
            np.cos(env_state['defender'][2]),
            np.sin(env_state['defender'][2])
        ])
        return nn_state
    
    
    
    def encode_helper(self, env_state):
        min_value = -self.size
        max_value = self.size

        def normalize_coordinate(coordinate, min_value, max_value):
            return (coordinate - min_value) / (max_value - min_value)

        goal_position = self.goal_position
        attacker_state = env_state['attacker']
        defender_state = env_state['defender']

        attacker_x_norm = normalize_coordinate(attacker_state[0], min_value, max_value)
        attacker_y_norm = normalize_coordinate(attacker_state[1], min_value, max_value)
        attacker_theta = attacker_state[2]

        defender_x_norm = normalize_coordinate(defender_state[0], min_value, max_value)
        defender_y_norm = normalize_coordinate(defender_state[1], min_value, max_value)
        defender_theta = defender_state[2]

        goal_x_norm = normalize_coordinate(goal_position[0], min_value, max_value)
        goal_y_norm = normalize_coordinate(goal_position[1], min_value, max_value)

        distance_attacker_goal = jnp.linalg.norm(jnp.array([goal_x_norm, goal_y_norm]) -jnp.array([attacker_x_norm, attacker_y_norm]))
        direction_attacker_goal = jnp.arctan2(goal_y_norm - attacker_y_norm, goal_x_norm - attacker_x_norm)

        angle_diff_attacker_goal = direction_attacker_goal - attacker_theta
        angle_diff_attacker_goal = (angle_diff_attacker_goal + jnp.pi) % (2 * jnp.pi) - jnp.pi
        facing_goal_attacker = jnp.cos(angle_diff_attacker_goal)

        #distance_attacker_defender = np.linalg.norm(np.array([defender_x_norm, defender_y_norm]) - np.array([attacker_x_norm, attacker_y_norm]))
        direction_attacker_defender = jnp.arctan2(defender_y_norm - attacker_y_norm, defender_x_norm - attacker_x_norm)

        angle_diff_attacker_defender = direction_attacker_defender - attacker_theta
        angle_diff_attacker_defender = (angle_diff_attacker_defender + jnp.pi) % (2 * jnp.pi) - jnp.pi
        facing_defender_attacker = jnp.cos(angle_diff_attacker_defender)

        angle_diff_defender_attacker = direction_attacker_defender - defender_theta + jnp.pi
        angle_diff_defender_attacker = (angle_diff_defender_attacker + jnp.pi) % (2 * jnp.pi) - jnp.pi
        facing_attacker_defender = jnp.cos(angle_diff_defender_attacker)

        dx = jnp.abs(attacker_x_norm - defender_x_norm)
        dy = jnp.abs(attacker_y_norm - defender_y_norm)
        dx_wrap = jnp.abs(dx - 1.0)
        dy_wrap = jnp.abs(dy - 1.0)
        min_dx = jnp.minimum(dx, dx_wrap)
        min_dy = jnp.minimum(dy, dy_wrap)
        wrapped_diff = jnp.array([min_dx, min_dy])
        distance_attacker_defender = jnp.linalg.norm(wrapped_diff)



        nn_state = jnp.array([attacker_x_norm, 
                             attacker_y_norm, 
                             jnp.cos(attacker_theta), 
                             jnp.sin(attacker_theta), 
                             defender_x_norm, 
                             defender_y_norm, 
                             jnp.cos(defender_theta), 
                             jnp.sin(defender_theta), 
                             distance_attacker_goal, 
                             facing_goal_attacker, 
                             distance_attacker_defender, 
                             facing_defender_attacker,
                             facing_attacker_defender
                            ])
        

        
        return nn_state

    
    def decode_helper(self, nn_states):
        min_value = -self.size
        max_value = self.size

        def unnormalize_coordinate(coordinate, min_value, max_value):
            return coordinate * (max_value - min_value) + min_value

        states = []
        for s in nn_states:

            attacker_x = unnormalize_coordinate(s[0], min_value, max_value)
            attacker_y = unnormalize_coordinate(s[1], min_value, max_value)
            attacker_theta = np.arctan2(s[3], s[2]) % (2 * np.pi) 

            defender_x = unnormalize_coordinate(s[4], min_value, max_value)
            defender_y = unnormalize_coordinate(s[5], min_value, max_value)
            defender_theta = np.arctan2(s[7], s[6]) % (2 * np.pi) 

            env_state = {
                'attacker': np.array([attacker_x, attacker_y, attacker_theta]),
                'defender': np.array([defender_x, defender_y, defender_theta])
            }
            states.append(env_state)
            
        return states


    def unconstrained_select_action(self, nn_state, params, policy_net, key, epsilon):
        if jax.random.uniform(key) < epsilon:
            return jax.random.choice(key, self.num_actions)
        else:
            probs = policy_net.apply(params, nn_state)
            return jax.random.choice(key, a=self.num_actions, p=probs)
    
    def get_legal_actions_mask(self, state, player):
        legal_actions_mask = []
        for action in range(self.num_actions):
            _, _, _, info = self.step(state, action,player, update_env=False)
            legal_actions_mask.append(int(info['is_legal']))
        return jnp.array(legal_actions_mask)

    def constrained_select_action(self, nn_state, policy_net, params, legal_actions_mask, key, epsilon):
        #print('FUNC')
        if jax.random.uniform(key) < epsilon:
            legal_actions_indices = jnp.arange(len(legal_actions_mask))[legal_actions_mask.astype(bool)]
            return jax.random.choice(key, legal_actions_indices)
        else:
            probs = policy_net.apply(params, nn_state, legal_actions_mask)
            #action = jax.random.categorical(key, probs)
            #print(nn_state)
            #print('probs', probs)
            action = jax.random.choice(key, a=self.num_actions, p=probs)
            #print('probs', probs, 'action', action)
            return action
        
    def constrained_deterministic_select_action(self, nn_state, policy_net, params, legal_actions_mask, key, epsilon):
        if jax.random.uniform(key) < epsilon:
            legal_actions_indices = jnp.arange(len(legal_actions_mask))[legal_actions_mask.astype(bool)]
            return jax.random.choice(key, legal_actions_indices)
        else:
            probs = policy_net.apply(params, nn_state, legal_actions_mask)
            action = jnp.argmax(probs)
            return action


    def deterministic_select_action(self, nn_state, params, policy_net):
        
        probs = policy_net.apply(params, nn_state)
        return jnp.argmax(probs)


    def get_closer(self, state, player):
        dists = []
        for action in range(self.num_actions):
            next_state, _, _, info = self.step(state, action,player, update_env=False)
            #get distance between players
            dist = np.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2])
            dists.append(dist)
        a = np.argmin(dists)
        return a




    


    def pad_and_mask(self, states, actions, action_masks, returns):
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        action_masks = np.array(action_masks)
        returns = np.array(returns)

        # Pad the states, actions, and returns
        padded_states = np.pad(states, ((0, self.max_steps - len(states)), (0, 0)), 'constant', constant_values=0)
        padded_actions = np.pad(actions, (0, self.max_steps  - len(actions)), 'constant', constant_values=0)
        padded_action_masks = np.pad(action_masks, ((0, self.max_steps  - len(action_masks)), (0,0)), 'constant', constant_values=0)
        padded_returns = np.pad(returns, (0, self.max_steps  - len(returns)), 'constant', constant_values=0)

        # Create a mask where the value is 1 for actual values and 0 for padding
        padding_mask = np.concatenate([np.ones(len(states)), np.zeros(self.max_steps  - len(states))])

        return padded_states, padded_actions, padded_action_masks, padded_returns, padding_mask






        
    def single_rollout(self,args):
        #print("env state" , self.state)
        game_type, params, policy_net, key, epsilon, gamma, render, for_q_value, one_step_reward, state = args

        if game_type != self.game_type:
            raise ValueError(f"game_type {game_type} does not match self.game_type {self.game_type}")


        states = {player: [] for player in self.players}
        actions = {player: [] for player in self.players}
        action_masks = {player: [] for player in self.players}
        rewards = {player: [] for player in self.players}
        padding_mask = {player: [] for player in self.players}

        wins = {'attacker': 0, 'defender': 0, 'draw': 0}
        done = False
        step = 0
        defender_wins = False
        attacker_wins = False
        defender_oob = False

        defender_no_legal_moves = False
        attacker_no_legal_moves = False


        if state == None:
            state = self.state
        #state = self.reset()
        else:
            self.state = state


        if for_q_value:
            #append 0 to rewards
            rewards['attacker'].append(one_step_reward)
            rewards['defender'].append(0)
            step = 1
        # else: 
        #     state = self.reset()
            
        nn_state = self.encode_helper(state)


        while not done and not defender_wins and not attacker_wins and step < self.max_steps:
            for player in self.players:
                states[player].append(nn_state)

                key, subkey = jax.random.split(key)


                if game_type == 'nash':
                    if player == 'attacker':
                        action = self.unconstrained_select_action(nn_state, params[player], policy_net,  subkey, epsilon)
                    else:
                        action = self.get_closer(state, player)
                    state, reward, done, info = self.step(state=state, action=action, player=player, update_env=True)
                    nn_state = self.encode_helper(state)
                    actions[player].append(action)
                    rewards[player].append(reward)
                    action_masks[player].append([1]*self.num_actions)


                elif game_type == 'stackelberg':
                    legal_actions_mask = self.get_legal_actions_mask(state, player)
                    if sum(legal_actions_mask) != 0:
                        if player == 'defender':
                            #action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)
                            action = self.get_closer(state, player)
                        elif player == 'attacker':
                            action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)

                            #action = self.constrained_deterministic_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)
                        #action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)
                        #print(nn_state, player, action)
                        #print(subkey)
                        action_masks[player].append(legal_actions_mask)
                        state, reward, done, info = self.step(state=state, action=action, player=player, update_env=True)
                        nn_state = self.encode_helper(state)
                        actions[player].append(action)
                        rewards[player].append(reward)
                    else: #case where a player has no legal moves
                        done = True
                        if player == 'defender': 
                            defender_no_legal_moves = True
                        elif player == 'attacker': 
                            attacker_no_legal_moves = True

                            
                            


                if player == 'attacker' and done:
                    if info['status'] == 'goal_reached':
                        attacker_wins = True
                        wins['attacker'] = 1
                    if info['status'] == 'attacker collided with defender':
                        defender_wins = True
                        wins['defender'] = 1
                    if info['status'] == 'out_of_bounds':
                        defender_wins = True
                        wins['defender'] = 1
                    if attacker_no_legal_moves:
                        defender_wins = True
                        wins['defender'] = 1
                        #rewards['attacker'][-1] = -200
                    break

                if player == 'defender' and done:
                    rewards['attacker'] = rewards['attacker'][:-1]
                    actions['attacker'] = actions['attacker'][:-1]
                    if defender_no_legal_moves:
                        #attacker_wins = True
                        #wins['attacker'] = 1
                        #rewards['attacker'][-1] = -20
                        pass
                    if info['status'] == 'defender collided with attacker':
                        defender_wins = True
                        wins['defender'] = 1
                        #rewards['attacker'][-1] = -1
                        pass
                    break

                # if step == self.max_steps - 1:
                #     done = True
                #     defender_wins = True
                #     wins['defender'] = 1
                #     rewards['attacker'][-1] = -1
                #     break


                

                

                # if done and player == 'defender' and info['is_legal'] == True: #only attacker can end the game, iterate one more time
                #     defender_wins = True
                #     wins['defender'] = 1

                # if (defender_wins and player == 'attacker'): #overwrite the attacker's last reward
                #     rewards['attacker'][-1] = -1
                #     done = True
                #     break

                # if done and player == 'defender' and info['is_legal'] == False: #only attacker can end the game, iterate one more time
                #     defender_oob = True

                # if (defender_oob and player == 'attacker'): #overwrite the attacker's last reward
                #     rewards['attacker'][-1] = 1
                #     done = True
                #     wins['attacker'] = 1
                #     break

                # if (done and player == 'attacker'): #break if attacker wins, game is over
                #     if info['is_legal']:
                #         attacker_wins = True
                #         wins['attacker'] = 1
                #     elif not info['is_legal']:
                #         defender_wins = True
                #         wins['defender'] = 1
                #     break
                
                
                
                
                if render:
                    self.render()



            step += 1
            #print(step)

        if not defender_wins and not attacker_wins:
            wins['draw'] = 1
            #rewards['attacker'][-1] = -1

        returns = {player: [] for player in self.players}

        for player in self.players:
            G = 0
            for r in reversed(rewards['attacker']):
                G = r + gamma * G
                returns[player].append(G)
            
            returns[player] = list(reversed(returns[player]))

        if not for_q_value:
            for player in self.players:
                states[player], actions[player], action_masks[player], returns[player], padding_mask[player] = self.pad_and_mask(states[player], actions[player], action_masks[player], returns[player])
    
        

        return states, actions, action_masks, returns, padding_mask, wins


    def single_rollout_eval(self,args):
        #print("env state" , self.state)
        game_type, params, policy_net, key, epsilon, gamma, render, for_q_value, one_step_reward, state = args

        if game_type != self.game_type:
            raise ValueError(f"game_type {game_type} does not match self.game_type {self.game_type}")


        states = {player: [] for player in self.players}
        actions = {player: [] for player in self.players}
        action_masks = {player: [] for player in self.players}
        rewards = {player: [] for player in self.players}
        padding_mask = {player: [] for player in self.players}

        wins = {'attacker': 0, 'defender': 0, 'draw': 0}
        done = False
        step = 0
        defender_wins = False
        attacker_wins = False
        defender_oob = False

        defender_no_legal_moves = False
        attacker_no_legal_moves = False


        if state == None:
            state = self.state
        #state = self.reset()
        else:
            self.state = state


        if for_q_value:
            #append 0 to rewards
            rewards['attacker'].append(one_step_reward)
            rewards['defender'].append(0)
            step = 1
        # else: 
        #     state = self.reset()
            
        nn_state = self.encode_helper(state)

        def_pos = []
        y_vals = [1.5, 0,-1.5,-100]
        subkeys = jax.random.split(key,len(y_vals))

        for i in range(len(y_vals)):
            pos = np.array([jax.random.uniform(subkeys[i], minval=-1, maxval=1),y_vals[i], np.pi/2])
            def_pos.append(pos)

        defender_index = 0
        state['defender'] = def_pos[0]
        while not done and not defender_wins and not attacker_wins and step < self.max_steps:

            for player in ['attacker']:
                if state['attacker'][1] < state['defender'][1]:
                    defender_index+=1
                    state['defender'] = def_pos[defender_index]
                states[player].append(nn_state)

                key, subkey = jax.random.split(key)


                if game_type == 'nash':
                    action = self.unconstrained_select_action(nn_state, params[player], policy_net,  subkey, epsilon)
                    state, reward, done, info = self.step(state=state, action=action, player=player, update_env=True)
                    nn_state = self.encode_helper(state)
                    actions[player].append(action)
                    rewards[player].append(reward)
                    action_masks[player].append([1]*self.num_actions)


                elif game_type == 'stackelberg':
                    legal_actions_mask = self.get_legal_actions_mask(state, player)
                    if sum(legal_actions_mask) != 0:
                        if player == 'defender':
                            action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)
                        elif player == 'attacker':
                            action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)

                            #action = self.constrained_deterministic_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)
                        #action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)
                        #print(nn_state, player, action)
                        #print(subkey)
                        action_masks[player].append(legal_actions_mask)
                        state, reward, done, info = self.step(state=state, action=action, player=player, update_env=True)
                        nn_state = self.encode_helper(state)
                        actions[player].append(action)
                        rewards[player].append(reward)
                    else: #case where a player has no legal moves
                        done = True
                        if player == 'defender': 
                            defender_no_legal_moves = True
                        elif player == 'attacker': 
                            attacker_no_legal_moves = True

                            
                            


                if player == 'attacker' and done:
                    if info['status'] == 'goal_reached':
                        attacker_wins = True
                        wins['attacker'] = 1
                    if info['status'] == 'attacker collided with defender':
                        defender_wins = True
                        wins['defender'] = 1
                    if info['status'] == 'out_of_bounds':
                        defender_wins = True
                        wins['defender'] = 1
                    if attacker_no_legal_moves:
                        defender_wins = True
                        wins['defender'] = 1
                        #rewards['attacker'][-1] = -200
                    break

                if player == 'defender' and done:
                    rewards['attacker'] = rewards['attacker'][:-1]
                    actions['attacker'] = actions['attacker'][:-1]
                    if defender_no_legal_moves:
                        #attacker_wins = True
                        #wins['attacker'] = 1
                        #rewards['attacker'][-1] = -20
                        pass
                    if info['status'] == 'defender collided with attacker':
                        #defender_wins = True
                        #wins['defender'] = 1
                        #rewards['attacker'][-1] = -1
                        pass
                    break

              
                
                
                
                if render:
                    self.render_eval(def_pos)



            step += 1
            #print(step)


        if not defender_wins and not attacker_wins:
            wins['draw'] = 1
            #rewards['attacker'][-1] = -1

        returns = {player: [] for player in self.players}

        for player in self.players:
            G = 0
            for r in reversed(rewards['attacker']):
                G = r + gamma * G
                returns[player].append(G)
            
            returns[player] = list(reversed(returns[player]))

        # if not for_q_value:
        #     for player in self.players:
        #         states[player], actions[player], action_masks[player], returns[player], padding_mask[player] = self.pad_and_mask(states[player], actions[player], action_masks[player], returns[player])
    
        

        return states, actions, action_masks, returns, padding_mask, wins



    def render_eval(self, def_pos, mode='human', close=False):
        """
        Render the environment for human viewing
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import io
        import imageio
        matplotlib.use('Agg')



        fig = plt.figure()
        plt.clf()
        plt.xlim([-self.size, self.size])
        plt.ylim([-self.size, self.size])

        # self.positions['attacker'].append(np.array(self.state['attacker'][:2].copy()))
        # self.positions['defender'].append(np.array(self.state['defender'][:2].copy()))

                # Append current positions and orientations to respective lists
        self.positions['attacker'].append((self.state['attacker'][:2].copy(), self.state['attacker'][2]))
        self.positions['defender'].append((self.state['defender'][:2].copy(), self.state['defender'][2]))



        #  # After adding the attacker, defender and goal
        # for player, positions in self.positions.items():
        #     color = 'b' if player == 'attacker' else 'r'
        #     for pos in positions:
        #         plt.plot(*pos, marker='o', markersize=2, color=color)

# Plot trails for each player
        for player, pos_orients in self.positions.items():
            color = 'b' if player == 'attacker' else 'r'
            for pos, orient in pos_orients:
                dx = 0.1 * np.cos(orient)
                dy = 0.1 * np.sin(orient)
                plt.arrow(*pos, dx, dy, color=color, head_width=0.1, head_length=0.1)
    

        attacker = plt.Circle((self.state['attacker'][0], self.state['attacker'][1]), 0.1, color='b', fill=True)
        
        
        for defender in def_pos:
            defender_plot = plt.Circle((defender[0], defender[1]), self.capture_radius, color='r', fill=True)
            defender_capture_radius = plt.Circle((defender[0], defender[1]), self.capture_radius+self.v_max, color='r', fill=False, linestyle='--')
            #plt.gca().add_artist(defender_capture_radius)
            plt.gca().add_artist(defender_plot)
        # defender = plt.Circle((self.state['defender'][0], self.state['defender'][1]), self.capture_radius, color='r', fill=True)
        # defender_capture_radius = plt.Circle((self.state['defender'][0], self.state['defender'][1]), self.capture_radius+self.v_max, color='r', fill=False, linestyle='--')



        # draw goal
        goal = plt.Circle((self.goal_position[0], self.goal_position[1]), self.goal_radius, color='g', fill=False)

        # draw obstacle
        plt.gca().add_artist(attacker)
        plt.gca().add_artist(goal)

        

        #print('saving')
        # save current figure to images list
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.images.append(np.array(imageio.v2.imread(buf)))
        plt.close()