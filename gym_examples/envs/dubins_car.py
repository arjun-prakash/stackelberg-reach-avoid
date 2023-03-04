import gym
from gym import spaces
import numpy as np
import jax

class DubinsCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.array([-4, -4, 0]), high=np.array([4, 4, 2*np.pi]), dtype=np.float64)        
        self.goal_position = np.array([0,0]) # position of the goal
        self.obstacle_position = np.array([-2,0]) # position of the obstacle
        self.obstacle_radius = 0.5 # radius of the obstacle
        self.state = np.array([0,0,0]) # position of the car
        self.min_distance_to_goal = 1 # minimum distance to goal to consider the task as done
        self.timestep = 1 # timestep in seconds
        self.v_max = 0.25 # maximum speed
        self.omega_max = 65 * np.pi/180  # maximum angular velocity (radians)
        self.images = []
        self.reward = 1
        #self.reset()
        


    def update_environment(self, next_state):
        self.state[0] = next_state[0]
        self.state[1] = next_state[1]
        self.state[2] = next_state[2]



    def step(self, state=None, action="",update_env=False):
        """
        Perform the action and return the next state, reward and done flag
        """



        v = self.v_max # speed of the car
        omega = self.omega_max # angular velocity of the car
        if action == 0: # turn left
            omega = -omega
        elif action == 2: # turn right
            omega = omega
        elif action == 1: # action 1 : straight
            omega = 0
        elif action ==3: # action 3: reverse left
            omega = -omega - np.pi
        elif action ==4: # action 4: reverse right
            omega = omega - np.pi


            





        if update_env:
            next_state = self.state.copy() #save copy of the state
            state = self.state
        else:
            next_state = state.copy() #save copy of the state

        #update the state

        next_state[2] += omega * self.timestep
        next_state[2] = (next_state[2]) % (2 * np.pi) 

        

        next_state[0] += v * np.cos(next_state[2]) * self.timestep
        next_state[1] += v * np.sin(next_state[2]) * self.timestep

        # print('state', state)
        # print('next_state', next_state)
        
            

            # update car position and orientation

        # check if the car is out of bounds
        if next_state[0] < self.observation_space.low[0] or next_state[0] > self.observation_space.high[0] or next_state[1] < self.observation_space.low[1] or next_state[1] > self.observation_space.high[1]:
            done = False
            reward = 0
            info = {'is_legal':False}
            #state[2] = ((next_state[2] - np.pi)) % (2 * np.pi) #np.random.uniform(low=-np.pi, high=np.pi)
            state = next_state #revert to previous state, but flip back



            if update_env:
                self.update_environment(next_state)
            return next_state, reward, done, info #make it end game, with -1



       
       # calculate distance to goal and obstacle
        dist_obstacle = np.linalg.norm(next_state[:2] - self.obstacle_position) - self.obstacle_radius
        if dist_obstacle < 0:
            done = False
            reward = -1
            info = {'is_legal':False}
            #state[2] = ((next_state[2] - np.pi)) % (2 * np.pi)#np.random.uniform(low=-np.pi, high=np.pi)
            next_state = state

            if update_env:
                self.update_environment(next_state)
            return next_state, reward, done, info #make it end game, with -1



            
        dist_goal = np.linalg.norm(next_state[:2] - self.goal_position) - self.min_distance_to_goal
        if dist_goal <= 0:
            state = next_state

            done = True
            reward = self.reward 
            info ={'is_legal':True}
            if update_env:
                self.update_environment(next_state)
            return next_state, reward, done, info #make it end game, with -1



        else:
            state = next_state
            reward = 0
            done = False
            info = {'is_legal':True}
            if update_env:
                self.update_environment(next_state)


            return next_state, reward, done, info #make it end game, with -1



    def sample(self, state, action, gamma):
        state_, reward, done, _ = self.step(state, action)

        next_rewards = []
        for a in range(self.action_space.n):
            _, reward_, _, _ = self.step(state_, a)
            next_rewards.append(reward_)
        expected_next_reward = np.max(next_rewards)

        return expected_next_reward 


    def get_reward(self, state):

        rewards = []
        legal = []
        for a in range(self.action_space.n):
            _, reward_, done, info = self.step(state, a)
            rewards.append(reward_)
            legal.append(info['is_legal'])

        if np.sum(legal) == 0:
            max_reward = -1
        else:
            max_reward = np.max(rewards)

        return max_reward 
    
    def get_reward2(self, state):

        rewards = []
        for a in range(self.action_space.n):
            _, reward_, done, _ = self.step(state, a)
            rewards.append(reward_)
        max_reward = np.mean(rewards)

        return max_reward 



    def sample_value_iter(self,X_batch, forward, params, gamma):
        y_hat = []
        for state in X_batch:
            state = self.decode_helper(state)
            values = []
            for action in range(self.action_space.n):
                state_, reward, done, info = self.step(state, action, update_env=False)
                state_ = self.encode_helper(state_)
                if done:
                    value = np.array([reward])
                    values.append(value)
                elif info['is_legal'] == False:
                    pass
                else:
                    value = reward + gamma*forward(X=state_, params=params)
                    values.append(value)

            if len(values) == 0:
                max_value = -1.
            else:
                max_value = np.max(values)
            y_hat.append(max_value)

        return np.array(y_hat)
    
    def decode_helper(self, encoded_state):
        x = encoded_state[0]
        y = encoded_state[1]
        theta = np.arctan2(encoded_state[3], encoded_state[2]) #y,x 
        theta = np.mod(theta, 2*np.pi)
        return np.array([x,y,theta])
    
    def encode_helper(self, state):
        x = state[0]
        y = state[1]
        theta = state[2]
        return np.array([x,y,np.cos(theta),np.sin(theta)])
        




            
            
    def reset(self):
        """
        Reset the environment and return the initial state
        """
        self.state = self.observation_space.sample()
        self.goal_position = self.goal_position
        self.obstacle_position = self.obstacle_position
        return self.state

    def set(self, x, y,theta):
        """
        Reset the environment and return the initial state
        """
        self.state = np.array([x, y, theta], dtype=self.observation_space.dtype)

        self.goal_position = self.goal_position
        self.obstacle_position = self.obstacle_position
        return self.state
        
    def render(self, mode='human', close=False):
        """
        Render the environment for human viewing
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import io
        import imageio
        matplotlib.use('Agg')
        from matplotlib import cm





        fig = plt.figure()
        plt.clf()
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])

        # draw car
        #car = plt.Circle((self.state[0], self.state[1]), 0.1, color='b', fill=True)


        # draw goal
        goal = plt.Circle((self.goal_position[0], self.goal_position[1]), self.min_distance_to_goal, color='g', fill=False)

        # draw obstacle
        obstacle = plt.Circle((self.obstacle_position[0], self.obstacle_position[1]), self.obstacle_radius, color='r', fill=False)
        plt.gca().add_artist(obstacle)
        plt.gca().add_artist(goal)
        #plt.gca().add_artist(car)

        arrow_len = self.v_max
        # Calculate arrow components
        arrow_dx = arrow_len * np.cos(self.state[2])
        arrow_dy = arrow_len * np.sin(self.state[2])



        plt.quiver(self.state[0], self.state[1], arrow_dx, arrow_dy, angles='xy', scale_units='xy', scale=1, width=0.01)
        plt.jet()


        

        #print('saving')
        # save current figure to images list
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.images.append(np.array(imageio.v2.imread(buf)))
        plt.close()

    def make_gif(self):
        import imageio

        imageio.mimsave('animation.gif', self.images, fps=30)



    def state_to_obs(self, state):
        """
        Convert the state to an observation format that the TileCoder class can understand

        Parameters
        ----------
        state : tuple of float
            The state to convert

        Returns
        -------
        obs : numpy array of shape (3,)
            The observation in the format (x, y, theta)
        """
        x, y, theta = state
        obs = np.array([x, y, theta])
        return obs



class TwoPlayerDubinsCarEnv(DubinsCarEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()


        self.players = ['defender', 'attacker']

        self.action_space = {'attacker':spaces.Discrete(3), 'defender':spaces.Discrete(3)}

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
            omega = - omega
        elif action == 2: # turn right
            omega = omega
        else: # action 1 : straight
            omega = 0

        

        if update_env:
            next_state = self.state.copy() #save copy of the state
            state = self.state.copy()
        else:
            next_state = state.copy() #save copy of the state

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
        if self.state['attacker'][0] < self.observation_space['attacker'].low[0] or self.state['attacker'][0] > self.observation_space['attacker'].high[0] or self.state['attacker'][1] < self.observation_space['attacker'].low[1] or self.state['attacker'][1] > self.observation_space[player].high[1]:
            print('out of bounds')
            reward = -self.reward
            done = True
            info = {'attacker': 'lost', 'defender':'won'}

            if update_env:
                self.state[player] = next_state

            return next_state, -reward, done, info
        

        if player == 'attacker':
            dist_capture = np.linalg.norm(next_state['attacker'][:2] - next_state['defender'][:2]) - self.capture_radius
        elif player == 'defender':
            dist_capture = np.linalg.norm(next_state['defender'][:2] - next_state['attacker'][:2]) - self.capture_radius
        
        if dist_capture < 0:
            reward = -self.reward
            done = True
            info = {'attacker': 'lost', 'defender':'won'}

            if update_env:
                self.state = next_state

            return next_state, -reward, done, info
       
        dist_goal = np.linalg.norm(next_state['attacker'][:2] - self.goal_position)
        if dist_goal < self.min_distance_to_goal:
            reward = self.reward
            done = True
            info = {'attacker': 'won', 'defender':'lost'}

            if update_env:
                self.state = next_state

            return next_state, reward, done, info
        
        else:
            state = next_state
            reward = 0
            done = False
            info = {}
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
                    next_env_state, reward, _, _ = self.step(env_state_, a_action, 'attacker')
                    next_nn_state = self.state_for_nn(next_env_state)
                    value = reward + gamma*forward(X=next_nn_state, params=params)
                    possible_actions.append([d_action, a_action, value[0]])

            pa = np.array(possible_actions)[:,2].reshape(3,3)
            #reward =  np.min(np.max(pa.T,axis=0))
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



        

