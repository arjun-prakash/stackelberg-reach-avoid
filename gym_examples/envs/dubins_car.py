import gym
from gym import spaces
import numpy as np
import jax

class DubinsCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(3)
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
        
        dist_goal = np.linalg.norm(next_state[:2] - self.goal_position) - self.min_distance_to_goal
        dreward = np.exp(-dist_goal)


            # update car position and orientation

        # # check if the car is out of bounds
        # if next_state[0] < self.observation_space.low[0] or next_state[0] > self.observation_space.high[0] or next_state[1] < self.observation_space.low[1] or next_state[1] > self.observation_space.high[1]:
        #     done = False
        #     reward = dreward
        #     info = {'is_legal':False}
        #     state[2] = ((next_state[2] - np.pi)) % (2 * np.pi) #np.random.uniform(low=-np.pi, high=np.pi)
        #     #state = next_state #revert to previous state, but flip back



        #     if update_env:
        #         self.update_environment(state)
        #     return state, reward, done, info #make it end game, with -1


                # wrap the car around if it goes out of bounds
        if next_state[0] < self.observation_space.low[0]:
            next_state[0] = self.observation_space.high[0]
        elif next_state[0] > self.observation_space.high[0]:
            next_state[0] = self.observation_space.low[0]

        if next_state[1] < self.observation_space.low[1]:
            next_state[1] = self.observation_space.high[1]
        elif next_state[1] > self.observation_space.high[1]:
            next_state[1] = self.observation_space.low[1]




       
       # calculate distance to goal and obstacle
        dist_obstacle = np.linalg.norm(next_state[:2] - self.obstacle_position) - self.obstacle_radius
        if dist_obstacle < 0:
            done = False
            reward = dreward
            info = {'is_legal':False}
            #state[2] = ((next_state[2] - np.pi)) % (2 * np.pi)#np.random.uniform(low=-np.pi, high=np.pi)
            next_state = state

            if update_env:
                self.update_environment(next_state)
            return next_state, reward, done, info #make it end game, with -1



            
        if dist_goal <= 0:
            state = next_state

            done = True
            reward = 1
            info ={'is_legal':True}
            if update_env:
                self.update_environment(next_state)
            return next_state, reward, done, info #make it end game, with -1



        else:
            state = next_state
            reward = dreward
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
    
    def encode_helper(self, state, norm=True):
        x = state[0]
        y = state[1]
        theta = state[2]


        if norm:
            x_norm = (x - self.observation_space.low[0]) / (self.observation_space.high[0] - self.observation_space.low[0])
            y_norm = (y - self.observation_space.low[1]) / (self.observation_space.high[1] - self.observation_space.low[1])

            # Normalize the coordinates and goal coordinates
            goal_x_norm = (self.goal_position[0] - self.observation_space.low[0]) / (self.observation_space.high[0] - self.observation_space.low[0])
            goal_y_norm = (self.goal_position[1] - self.observation_space.low[1]) / (self.observation_space.high[1] - self.observation_space.low[1])

            # Compute the normalized distance to the goal
            distance_to_goal = np.linalg.norm(np.array([goal_x_norm, goal_y_norm]) - np.array([x_norm, y_norm]))

            # Compute the direction to the goal
            direction_to_goal = np.arctan2(goal_y_norm - y_norm, goal_x_norm - x_norm)

            # Compute the angle between the agent's orientation and the direction to the goal
            angle_diff = direction_to_goal - theta

            # Normalize the angle difference to the range [-pi, pi]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

            # Compute the cosine of the angle difference
            facing_goal = np.cos(angle_diff)

            return np.array([x_norm, y_norm, np.cos(theta), np.sin(theta), distance_to_goal, facing_goal])

        else:
            return np.array([x,y,np.cos(theta),np.sin(theta)])
        




            
            
    def reset(self):
        """
        Reset the environment and return the initial state
        """
        illegal = True
        self.goal_position = self.goal_position
        self.obstacle_position = self.obstacle_position

        while illegal:
            self.state = self.observation_space.sample()
            dist_obstacle = np.linalg.norm(self.state[:2] - self.obstacle_position) - self.obstacle_radius
            dist_goal = np.linalg.norm(self.state[:2] - self.goal_position) - self.min_distance_to_goal

            if dist_obstacle > 0 and dist_goal > 0:
                illegal = False

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

        imageio.mimsave('animation_pg.gif', self.images, fps=30)



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






        

