

import gym
from gym import spaces
import numpy as np
import jax
import copy
import jax.numpy as jnp
import haiku as hk
import optax


from envs.dubins_car import DubinsCarEnv

class TwoPlayerDubinsCarEnv(DubinsCarEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, game_type, num_actions, size, reward, max_steps, init_defender_position, init_attacker_position, capture_radius, goal_position, goal_radius, timestep, v_max, omega_max):
        super().__init__()

        self.game_type = game_type
        self.players = ['attacker', 'defender']
        self.num_actions = num_actions #3
        self.action_space = {'attacker':spaces.Discrete(self.num_actions), 'defender':spaces.Discrete(self.num_actions)}

        self.size = size #4
        self.reward = reward #1
        self.max_steps = max_steps# 50

        self.observation_space= {'attacker':spaces.Box(low=np.array([-self.size, -self.size, 0]), high=np.array([self.size,self.size , 2*np.pi]), dtype=np.float32), 
                                'defender':spaces.Box(low=np.array([-self.size, -self.size, 0]), high=np.array([self.size, self.size, 2*np.pi]), dtype=np.float32)}



        self.init_defender_position = init_defender_position #np.array([0,0,0])
        self.init_attacker_position = init_attacker_position #np.array([2,2,0])
        self.state = {'attacker': np.array([self.init_attacker_position]), 'defender':np.array(self.init_defender_position)}
        self.capture_radius = capture_radius #0.5 # radius of the obstacle

        self.goal_position = np.array(goal_position) # position of the goal
        self.goal_radius = goal_radius # minimum distance to goal to consider the task as done

        self.timestep = timestep # timestep in seconds
        self.v_max = v_max # maximum speed
        self.omega_max = omega_max * np.pi/180  # maximum angular velocity (radians)
        self.images = []
        self.positions = {'attacker': [], 'defender': []}
        

    def reset(self, key=None):
        """
        Reset the environment and return the initial state
        """
        if key is not None: #move attacker lefit or right
            attacker_position = self.init_attacker_position
            perturbation = jax.random.choice(key, a=np.array([3,2,1]))
            attacker_position[1] = perturbation
            self.state['attacker'] = np.array(attacker_position, dtype=self.observation_space['defender'].dtype)
        else:
            self.state['attacker'] = np.array(self.init_attacker_position, dtype=self.observation_space['attacker'].dtype)


        self.state['defender'] = np.array(self.init_defender_position, dtype=self.observation_space['defender'].dtype)

        illegal = True
        epsilon = 0.25

        while illegal:
           # self.state['attacker'] = self.observation_space['attacker'].sample()
            #self.state['defender'] = self.observation_space['defender'].sample()

            dist_capture = np.linalg.norm(self.state['attacker'][:2] - self.state['defender'][:2]) - self.capture_radius - 1
            dist_goal = np.linalg.norm(self.state['attacker'][:2] - self.goal_position) - self.goal_radius - 1

            if dist_capture > 0 and dist_goal > 0:
                illegal = False

        return self.state


        self.goal_position = self.goal_position
        return self.state
    
    def set(self, ax, ay, atheta, dx=0., dy=0., dtheta=0.):
        """
        Reset the environment and return the initial state
        """
        self.state['attacker'] = np.array([ax, ay, atheta], dtype=self.observation_space['attacker'].dtype)
        self.state['defender'] = np.array([dx, dy, dtheta], dtype=self.observation_space['defender'].dtype)

        self.goal_position = self.goal_position
        return self.state


    def step(self, state=None, action=None, player=None, update_env=False):
        """
        Perform the action and return the next state, reward and done flag
        action: dict{attacker:int, defender:int}
        """


          
        v = self.v_max # speed of the car
        
        omega = self.omega_max # angular velocity of the car
        if action == 0: # turn left
            omega = -omega
        elif action == 2: # turn right
            omega = omega
        
        elif action == 1: # action 1 : straight
            omega = 0
        else:
            #reverse
            omega = -np.pi
            v = self.v_max

        

        if update_env:
            next_state = self.state.copy() #save copy of the state
            state = self.state.copy()
        else:
            next_state = copy.deepcopy(state)  # save deep copy of the state
            

        #update the state

        next_state[player][2] += omega * self.timestep
        next_state[player][2] = (next_state[player][2]) % (2 * np.pi) 


        next_state[player][0] += v * np.cos(next_state[player][2]) * self.timestep
        next_state[player][1] += v * np.sin(next_state[player][2]) * self.timestep

        dist_goal = np.linalg.norm(next_state['attacker'][:2] - self.goal_position)
        #reward = np.exp(-dist_goal)
        max_distance = np.sqrt(self.size**2 + self.size**2)

        #reward = 1/(dist_goal**2)
        #reward = -((dist_goal - self.goal_radius)**2)/ max_distance**2
        #reward = -((dist_goal - self.goal_radius)**2)
        dist_capture = np.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2]) 

        reward = -((dist_capture)**2) #+ (dist_goal**2)
        




        # if next_state[player][0] < self.observation_space[player].low[0]:
        #     next_state[player][0] = self.observation_space[player].high[0]
        # elif next_state[player][0] > self.observation_space[player].high[0]:
        #     next_state[player][0] = self.observation_space[player].low[0]

        # if next_state[player][1] < self.observation_space[player].low[1]:
        #     next_state[player][1] = self.observation_space[player].high[1]
        # elif next_state[player][1] > self.observation_space[player].high[1]:
        #     next_state[player][1] = self.observation_space[player].low[1]


        # out_of_bounds = False

        # # Check x-boundary
        # if next_state[player][0] <= self.observation_space[player].low[0] or next_state[player][0] >= self.observation_space[player].high[0]:
        #     out_of_bounds = True
        #     next_state[player][2] = (np.pi + next_state[player][2]) % (2 * np.pi)

        # # Check y-boundary
        # if next_state[player][1] <= self.observation_space[player].low[1] or next_state[player][1] >= self.observation_space[player].high[1]:
        #     out_of_bounds = True
        #     next_state[player][2] = (np.pi + next_state[player][2]) % (2 * np.pi)

        out_of_bounds = False

        # Check and handle x-boundary
        if next_state[player][0] <= self.observation_space[player].low[0]:
            # Reflect position and orientation if it crosses the left boundary
            next_state[player][0] = self.observation_space[player].low[0] + (self.observation_space[player].low[0] - next_state[player][0])
            next_state[player][2] = -next_state[player][2]
            out_of_bounds = True
        elif next_state[player][0] >= self.observation_space[player].high[0]:
            # Reflect position and orientation if it crosses the right boundary
            next_state[player][0] = self.observation_space[player].high[0] - (next_state[player][0] - self.observation_space[player].high[0])
            next_state[player][2] = -next_state[player][2]
            out_of_bounds = True

        # Check and handle y-boundary
        if next_state[player][1] <= self.observation_space[player].low[1]:
            # Reflect position and orientation if it crosses the bottom boundary
            next_state[player][1] = self.observation_space[player].low[1] + (self.observation_space[player].low[1] - next_state[player][1])
            next_state[player][2] = np.pi - next_state[player][2]
            out_of_bounds = True
        elif next_state[player][1] >= self.observation_space[player].high[1]:
            # Reflect position and orientation if it crosses the top boundary
            next_state[player][1] = self.observation_space[player].high[1] - (next_state[player][1] - self.observation_space[player].high[1])
            next_state[player][2] = np.pi - next_state[player][2]
            out_of_bounds = True

        # Normalize the angle to keep it within [0, 2*pi)
        next_state[player][2] = next_state[player][2] % (2 * np.pi)


        if out_of_bounds:
            reward = reward
            # next_state[player][0] = state[player][0]
            # next_state[player][1] = state[player][1]
            done = False
            if player == 'defender':
                is_legal = True
            else:
                is_legal = False

            info = {'player': player, 'is_legal':is_legal, 'status':'out_of_bounds'}
            # print('oob')
            # print(self.state)
            if update_env:
                self.state = next_state

            return next_state, reward, done, info


        
        


        #self.reward = np.exp(-dist_goal) - np.exp(-dist_capture)
        

        if self.game_type == 'nash':
            done_on_capture = True
        else:
            done_on_capture = False #no constraints
      
        if player == 'attacker':
            if dist_capture < self.capture_radius:
                info = {'player': player, 'is_legal':False, 'status':'attacker collided with defender'}
                done = done_on_capture #should it be false?
                next_state = state.copy()
                reward = 100 #reward

                if update_env:
                    self.state = next_state

                return next_state, reward, done, info

        elif player == 'defender':
            if dist_capture < self.capture_radius:
                info = {'player': player, 'is_legal':True, 'status':'defender collided with attacker'}
                done = done_on_capture #True
                reward = 100 #reward
                #next_state = state.copy()


                if update_env:
                    self.state = next_state

                return next_state, reward, done, info
       
        if dist_goal < self.goal_radius:
            reward = -100 #reward
            done = True
            info = {'player': player, 'is_legal':True, 'status':'goal_reached'}

            if update_env:
                self.state = next_state

            return next_state, reward, done, info
        
        else:
            reward = reward
            done = False
            info = {'player': player, 'is_legal':True, 'status':'in_progress'}
            if update_env:
                self.state = next_state

            return next_state, reward, done, info #make it end game, with -1





    def sample(self, state, action,player,gamma):
        state_, reward, done, _ = self.state_action_step(state, action, player)

        next_rewards = []

        for player in self.players:
            for a in range(self.action_space[player].n):
                _, reward_, _, _ = self.state_action_step(state, a, player)
                next_rewards.append(reward_)
            expected_next_reward = np.mean(np.array(next_rewards), axis=0)

        return reward + gamma*expected_next_reward





    def state_for_env(self,nn_state):
        """convert the state from the neural network np.array representation to the environment representation dict"""

        env_state = {'attacker':np.array([nn_state[0], nn_state[1], np.arctan2(nn_state[3], nn_state[2])]), 'defender':np.array([nn_state[4], nn_state[5], np.arctan2(nn_state[7], nn_state[6])])}
        return env_state

    def state_for_nn(self,env_state):
        """convert the state from the environment (dict) representation to the neural network representation np.array"""
        nn_state = np.array([env_state['attacker'][0], env_state['attacker'][1], np.cos(env_state['attacker'][2]), np.sin(env_state['attacker'][2]), env_state['defender'][0], env_state['defender'][1], np.cos(env_state['defender'][2]), np.sin(env_state['defender'][2])])
        return nn_state
    



    def sample_value_iter(self,X_batch, forward, params, gamma):
        y_hat = []
        for state in X_batch:
            env_state = self.decode_state_big(state)
            possible_actions = []
            for d_action in range(self.action_space['defender'].n):
                env_state_ , reward, done, info = self.step(env_state, d_action, 'defender')
                for a_action in range(self.action_space['attacker'].n):
                    next_env_state, attacker_reward, attacker_done, attacker_info = self.step(env_state_, a_action, 'attacker')

                    if attacker_info['is_legal'] == False:
                        possible_actions.append([d_action, a_action, attacker_reward]) #reward -1 if eaten, 0 if ou
                    elif attacker_done:
                        possible_actions.append([d_action, a_action, attacker_reward]) #reward 1
                        print('attaker reward', attacker_reward)

                    else:
                        next_nn_state = self.encode_state_big(next_env_state)
                        value = attacker_reward + gamma*forward(X=next_nn_state, params=params) #check bug
                        possible_actions.append([d_action, a_action, value[0]])

                    
            pa = np.array(possible_actions)[:,2].reshape(self.num_actions,self.num_actions)
            if np.all(pa == -1):
                best_value = -1 #defender wins

            else:
            #   reward =  np.min(np.max(pa.T,axis=0))
                best_attacker_moves = np.argmax(pa.T,axis=0)
                best_defender_move =  np.argmin(np.max(pa.T,axis=0))
                best_attacker_move = best_attacker_moves[best_defender_move]
                best_value = pa[best_defender_move][best_attacker_move]


            y_hat.append(best_value)

        return np.array(y_hat)


            


    def render(self, mode='human', close=False):
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
        defender = plt.Circle((self.state['defender'][0], self.state['defender'][1]), self.capture_radius, color='r', fill=True)
        defender_capture_radius = plt.Circle((self.state['defender'][0], self.state['defender'][1]), self.capture_radius+self.v_max, color='r', fill=False, linestyle='--')



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



    def get_attacker_max_reward(self, state):
        max_reward = -np.inf
        for d_action in range(self.action_space['defender'].n):
            state_, _, _, info_d = self.step(state, d_action, 'defender')
            for a_action in range(self.action_space['attacker'].n):
                _, reward, _, info_a = self.step(state_, a_action, 'attacker')
                if info_a['is_legal']:
                    max_reward = max(max_reward, reward)
        return max_reward


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

        distance_attacker_goal = np.linalg.norm(np.array([goal_x_norm, goal_y_norm]) - np.array([attacker_x_norm, attacker_y_norm]))
        direction_attacker_goal = np.arctan2(goal_y_norm - attacker_y_norm, goal_x_norm - attacker_x_norm)

        angle_diff_attacker_goal = direction_attacker_goal - attacker_theta
        angle_diff_attacker_goal = (angle_diff_attacker_goal + np.pi) % (2 * np.pi) - np.pi
        facing_goal_attacker = np.cos(angle_diff_attacker_goal)

        #distance_attacker_defender = np.linalg.norm(np.array([defender_x_norm, defender_y_norm]) - np.array([attacker_x_norm, attacker_y_norm]))
        direction_attacker_defender = np.arctan2(defender_y_norm - attacker_y_norm, defender_x_norm - attacker_x_norm)

        angle_diff_attacker_defender = direction_attacker_defender - attacker_theta
        angle_diff_attacker_defender = (angle_diff_attacker_defender + np.pi) % (2 * np.pi) - np.pi
        facing_defender_attacker = np.cos(angle_diff_attacker_defender)

        angle_diff_defender_attacker = direction_attacker_defender - defender_theta + np.pi
        angle_diff_defender_attacker = (angle_diff_defender_attacker + np.pi) % (2 * np.pi) - np.pi
        facing_attacker_defender = np.cos(angle_diff_defender_attacker)

        dx = np.abs(attacker_x_norm - defender_x_norm)
        dy = np.abs(attacker_y_norm - defender_y_norm)
        dx_wrap = np.abs(dx - 1.0)
        dy_wrap = np.abs(dy - 1.0)
        min_dx = np.minimum(dx, dx_wrap)
        min_dy = np.minimum(dy, dy_wrap)
        wrapped_diff = np.array([min_dx, min_dy])
        distance_attacker_defender = np.linalg.norm(wrapped_diff)



        nn_state = np.array([attacker_x_norm, 
                             attacker_y_norm, 
                             np.cos(attacker_theta), 
                             np.sin(attacker_theta), 
                             defender_x_norm, 
                             defender_y_norm, 
                             np.cos(defender_theta), 
                             np.sin(defender_theta), 
                             distance_attacker_goal, 
                             facing_goal_attacker, 
                             distance_attacker_defender, 
                             facing_defender_attacker,
                             facing_attacker_defender
                            ])
        

        
        return nn_state

    
    def decode_helper(self, nn_state):
        min_value = -self.size
        max_value = self.size

        def unnormalize_coordinate(coordinate, min_value, max_value):
            return coordinate * (max_value - min_value) + min_value

        attacker_x = unnormalize_coordinate(nn_state[0], min_value, max_value)
        attacker_y = unnormalize_coordinate(nn_state[1], min_value, max_value)
        attacker_theta = np.arctan2(nn_state[3], nn_state[2]) % (2 * np.pi) 

        defender_x = unnormalize_coordinate(nn_state[4], min_value, max_value)
        defender_y = unnormalize_coordinate(nn_state[5], min_value, max_value)
        defender_theta = np.arctan2(nn_state[7], nn_state[6]) % (2 * np.pi) 

        env_state = {
            'attacker': np.array([attacker_x, attacker_y, attacker_theta]),
            'defender': np.array([defender_x, defender_y, defender_theta])
        }
        return env_state


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
        if jax.random.uniform(key) < epsilon:
            legal_actions_indices = jnp.arange(len(legal_actions_mask))[legal_actions_mask.astype(bool)]
            return jax.random.choice(key, legal_actions_indices)
        else:
            probs = policy_net.apply(params, nn_state, legal_actions_mask)
            #action = jax.random.categorical(key, probs)
            action = jax.random.choice(key, a=self.num_actions, p=probs)
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
        game_type, params, policy_net, key, epsilon, gamma, render, for_q_value, one_step_reward = args

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


        state = self.state
        #state = self.reset()

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
                        rewards['defender'][-1] = 100
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
            for r in reversed(rewards['defender']):
                G = r + gamma * G
                returns[player].append(G)
            
            returns[player] = list(reversed(returns[player]))

        if not for_q_value:
            for player in self.players:
                states[player], actions[player], action_masks[player], returns[player], padding_mask[player] = self.pad_and_mask(states[player], actions[player], action_masks[player], returns[player])
    
        

        return states, actions, action_masks, returns, padding_mask, wins



    def single_rollout_round_robin(self,args):
        #print("env state" , self.state)
        game_type, params_a, params_d, policy_net_stackelberg, policy_net_nash, player_types, key, epsilon, gamma, render, for_q_value, one_step_reward = args

        # if game_type != self.game_type:
        #     raise ValueError(f"game_type {game_type} does not match self.game_type {self.game_type}")


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


        state = self.state
        #state = self.reset()

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


                if player_types[player] == 'nash':
                    if player == 'defender':
                        action = self.unconstrained_select_action(nn_state, params_d[player], policy_net_nash,  subkey, epsilon)
                    elif player == 'attacker':
                        action = self.unconstrained_select_action(nn_state, params_a[player], policy_net_nash,  subkey, epsilon)
                    state, reward, done, info = self.step(state=state, action=action, player=player, update_env=True)
                    nn_state = self.encode_helper(state)
                    actions[player].append(action)
                    rewards[player].append(reward)
                    action_masks[player].append([1]*self.num_actions)


                elif player_types[player] == 'stackelberg':
                    legal_actions_mask = self.get_legal_actions_mask(state, player)
                    if sum(legal_actions_mask) != 0:
                        if player == 'defender':
                            action = self.constrained_select_action(nn_state, policy_net_stackelberg, params_d[player], legal_actions_mask, subkey, epsilon)
                        elif player == 'attacker':
                            action = self.constrained_select_action(nn_state, policy_net_stackelberg, params_a[player], legal_actions_mask, subkey, epsilon)

                            #action = self.constrained_deterministic_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)
                        #action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)

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
                        rewards['defender'][-1] = 100
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
            for r in reversed(rewards['defender']):
                G = r + gamma * G
                returns[player].append(G)
            
            returns[player] = list(reversed(returns[player]))

        if not for_q_value:
            for player in self.players:
                states[player], actions[player], action_masks[player], returns[player], padding_mask[player] = self.pad_and_mask(states[player], actions[player], action_masks[player], returns[player])
    
        

        return states, actions, action_masks, returns, padding_mask, wins, step