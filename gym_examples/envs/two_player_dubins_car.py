

import gym
from gym import spaces
import numpy as np
import jax
import copy
import jax.numpy as jnp


from envs.dubins_car import DubinsCarEnv

class TwoPlayerDubinsCarEnv(DubinsCarEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()


        self.players = ['defender', 'attacker']
        self.num_actions = 4
        self.action_space = {'attacker':spaces.Discrete(self.num_actions), 'defender':spaces.Discrete(self.num_actions)}

        self.size = 4
        self.reward = 1

        self.observation_space= {'attacker':spaces.Box(low=np.array([-self.size, -self.size, 0]), high=np.array([self.size,self.size , 2*np.pi]), dtype=np.float32), 
                                'defender':spaces.Box(low=np.array([-self.size, -self.size, 0]), high=np.array([self.size, self.size, 2*np.pi]), dtype=np.float32)}



        self.state = {'attacker': np.array([0,0,0]), 'defender':np.array([0,0,np.pi])}

        self.goal_position = np.array([3,0]) # position of the goal
        self.capture_radius = 0.5 # radius of the obstacle



        self.min_distance_to_goal = 1 # minimum distance to goal to consider the task as done
        self.min_distance_to_obstacle = 0.1 # minimum distance to obstacle to consider the task as done

        self.timestep = 1 # timestep in seconds
        self.v_max = 0.25 # maximum speed
        self.omega_max = 65 * np.pi/180  # maximum angular velocity (radians)
        self.images = []
        

    def reset(self):
        """
        Reset the environment and return the initial state
        """
        self.state['attacker'] = self.observation_space['attacker'].sample()
        self.state['defender'] = np.array([0, 0., 0.], dtype=self.observation_space['defender'].dtype)
        #self.car_position['defender'] = np.array([2,2,2])


        self.goal_position = self.goal_position
        return self.state
    
    def set(self, ax, ay, atheta, dx, dy, dtheta):
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
        self.reward = np.exp(-dist_goal)





        # # check if the car is out of bounds
        # if self.car_position[player][0] < self.observation_space[player].low[0] or self.car_position[player][0] > self.observation_space[player].high[0] or self.car_position[player][1] < self.observation_space[player].low[1] or self.car_position[player][1] > self.observation_space[player].high[1]:
        #     print('out of bounds')
        #     return self.car_position, -10, True, {}

        # check if the car is out of bounds
        # if next_state['attacker'][0] < self.observation_space['attacker'].low[0] or next_state['attacker'][0] > self.observation_space['attacker'].high[0] or next_state['attacker'][1] < self.observation_space['attacker'].low[1] or next_state['attacker'][1] > self.observation_space[player].high[1]:
        #     #print('out of bounds')
        #     reward =  self.reward #or 0
        #     done = False
        #     info = {'player': player,'is_legal':False, 'status':'out_of_bounds'}

        #     next_state = state.copy()

        #     # if update_env:
        #     #     self.state = state

        #     return state, reward, done, info

        if next_state[player][0] < self.observation_space[player].low[0]:
            next_state[player][0] = self.observation_space[player].high[0]
        elif next_state[player][0] > self.observation_space[player].high[0]:
            next_state[player][0] = self.observation_space[player].low[0]

        if next_state[player][1] < self.observation_space[player].low[1]:
            next_state[player][1] = self.observation_space[player].high[1]
        elif next_state[player][1] > self.observation_space[player].high[1]:
            next_state[player][1] = self.observation_space[player].low[1]


        
        

        dist_capture = np.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2]) - self.capture_radius
      
        
        if dist_capture < 0:


            if player == 'attacker':
                info = {'player': player, 'is_legal':False, 'status':'cannot move into defender'}
                done = False
                next_state = state.copy()
                reward = -1

            else: #defender eats attacker
                info = {'player': player, 'is_legal':True, 'status':'eaten'}
                done = True
                reward = -1 # -self.reward


            if update_env:
                self.state = next_state

            return next_state, reward, done, info
       
        if dist_goal < self.min_distance_to_goal:
            reward = 1
            done = True
            info = {'player': player, 'is_legal':True, 'status':'goal_reached'}

            if update_env:
                self.state = next_state

            return next_state, reward, done, info
        
        else:
            reward = self.reward
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
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])

    

        attacker = plt.Circle((self.state['attacker'][0], self.state['attacker'][1]), 0.1, color='b', fill=True)
        defender = plt.Circle((self.state['defender'][0], self.state['defender'][1]), self.capture_radius, color='r', fill=True)


        # draw goal
        goal = plt.Circle((self.goal_position[0], self.goal_position[1]), self.min_distance_to_goal, color='g', fill=False)

        # draw obstacle
        plt.gca().add_artist(attacker)
        plt.gca().add_artist(defender)
        plt.gca().add_artist(goal)

        #print('saving')
        # save current figure to images list
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.images.append(np.array(imageio.v2.imread(buf)))
        plt.close()

    def make_gif(self):
        import imageio

        imageio.mimsave('two_player_animation_pg.gif', self.images, fps=30)



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

        distance_attacker_defender = np.linalg.norm(np.array([defender_x_norm, defender_y_norm]) - np.array([attacker_x_norm, attacker_y_norm]))
        direction_attacker_defender = np.arctan2(defender_y_norm - attacker_y_norm, defender_x_norm - attacker_x_norm)

        angle_diff_attacker_defender = direction_attacker_defender - attacker_theta
        angle_diff_attacker_defender = (angle_diff_attacker_defender + np.pi) % (2 * np.pi) - np.pi
        facing_defender_attacker = np.cos(angle_diff_attacker_defender)

        angle_diff_defender_attacker = direction_attacker_defender - defender_theta + np.pi
        angle_diff_defender_attacker = (angle_diff_defender_attacker + np.pi) % (2 * np.pi) - np.pi
        facing_attacker_defender = np.cos(angle_diff_defender_attacker)

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
    