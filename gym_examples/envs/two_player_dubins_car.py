

import gym
from gym import spaces
import numpy as np
import jax
import copy

from envs.dubins_car import DubinsCarEnv

class TwoPlayerDubinsCarEnv(DubinsCarEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()


        self.players = ['defender', 'attacker']
        self.num_actions = 3
        self.action_space = {'attacker':spaces.Discrete(self.num_actions), 'defender':spaces.Discrete(self.num_actions)}

        self.size = 4
        self.reward = 1

        self.observation_space= {'attacker':spaces.Box(low=np.array([-self.size, -self.size, 0]), high=np.array([self.size,self.size , 2*np.pi]), dtype=np.float32), 
                                'defender':spaces.Box(low=np.array([-self.size, -self.size, 0]), high=np.array([self.size, self.size, 2*np.pi]), dtype=np.float32)}



        self.state = {'attacker': np.array([0,0,0]), 'defender':np.array([0,0,0])}

        self.goal_position = np.array([0,0]) # position of the goal
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
        self.state['defender'] = self.observation_space['defender'].sample()
        #self.car_position['defender'] = np.array([2,2,2])


        self.goal_position = np.array([0,0]) 
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




        # # check if the car is out of bounds
        # if self.car_position[player][0] < self.observation_space[player].low[0] or self.car_position[player][0] > self.observation_space[player].high[0] or self.car_position[player][1] < self.observation_space[player].low[1] or self.car_position[player][1] > self.observation_space[player].high[1]:
        #     print('out of bounds')
        #     return self.car_position, -10, True, {}

        # check if the car is out of bounds
        if next_state['attacker'][0] < self.observation_space['attacker'].low[0] or next_state['attacker'][0] > self.observation_space['attacker'].high[0] or next_state['attacker'][1] < self.observation_space['attacker'].low[1] or next_state['attacker'][1] > self.observation_space[player].high[1]:
            #print('out of bounds')
            reward = -self.reward
            done = False
            info = {'player': player,'is_legal':False, 'status':'out_of_bounds'}

            next_state = state.copy()

            # if update_env:
            #     self.state = state

            return state, -reward, done, info
        

        dist_capture = np.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2]) - self.capture_radius
      
        
        if dist_capture < 0:
            reward = -self.reward
            done = False


            if player == 'attacker':
                info = {'player': player, 'is_legal':False}
                done = False
                next_state = state.copy()

            else: #defender eats attacker
                info = {'player': player, 'is_legal':True, 'status':'eaten'}
                done = True


            if update_env:
                self.state = next_state

            return next_state, reward, done, info
       
        dist_goal = np.linalg.norm(next_state['attacker'][:2] - self.goal_position)
        if dist_goal < self.min_distance_to_goal:
            reward = self.reward
            done = True
            info = {'player': player, 'is_legal':True, 'status':'goal_reached'}

            if update_env:
                self.state = next_state

            return next_state, reward, done, info
        
        else:
            reward = 0
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
            env_state = self.state_for_env(state)
            possible_actions = []
            for d_action in range(self.action_space['defender'].n):
                env_state_ , reward, done, info = self.step(env_state, d_action, 'defender')
                for a_action in range(self.action_space['attacker'].n):
                    next_env_state, reward, _, attacker_info = self.step(env_state_, a_action, 'attacker')

                    if attacker_info['is_legal'] == False:
                        possible_actions.append([d_action, a_action, 0])
                    else:
                        next_nn_state = self.state_for_nn(next_env_state)
                        value = reward + gamma*forward(X=next_nn_state, params=params)
                        possible_actions.append([d_action, a_action, value[0]])

                    

            pa = np.array(possible_actions)[:,2].reshape(self.num_actions,self.num_actions)
            if np.all(pa == 0):
                best_value = 0 #defender wins

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

        imageio.mimsave('two_player_animation.gif', self.images, fps=30)