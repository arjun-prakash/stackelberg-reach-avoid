import gym
from gym import spaces
import numpy as np
import jax

class DubinsCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([-4, -4]), high=np.array([4, 4]), dtype=np.float64)
        self.goal_position = np.array([0, 0])  # position of the goal
        self.obstacle_position = np.array([-2, 0])  # position of the obstacle
        self.obstacle_radius = 0.5  # radius of the obstacle
        front_point = np.array([0, 0])  # [x,y] front point of the car
        rear_point = np.array([-0.1, -0.1])  # [x,y] rear point of the car
        self.state = np.column_stack((front_point, rear_point))  # 2x2 position matrix
        self.wheelbase = 0.25  # distance between front and rear wheels


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



    def step(self, state=None, action="", update_env=False):
        v = self.v_max  # speed of the car
        omega = self.omega_max  # angular velocity of the car

        if action == 0:  # turn left
            omega = -omega
        elif action == 2:  # turn right
            omega = omega
        elif action == 1:  # action 1 : straight
            omega = 0
        elif action == 3:  # action 3: reverse left
            omega = -omega - np.pi
        elif action == 4:  # action 4: reverse right
            omega = omega - np.pi

        rotation_matrix = np.array([[np.cos(omega * self.timestep), -np.sin(omega * self.timestep)],
                                    [np.sin(omega * self.timestep), np.cos(omega * self.timestep)]])

        if update_env:
            next_state = self.state.copy()  # save copy of the state
            state = self.state
        else:
            next_state = state.copy()  # save copy of the state

        # update the state
        # next_state[:, 0] = rotation_matrix @ (next_state[:, 0] + np.array([v * self.timestep, v * self.timestep]))
        # rear_direction = np.array([np.cos(omega * self.timestep + np.pi), np.sin(omega * self.timestep + np.pi)])
        # next_state[:, 1] = next_state[:, 0] + self * rear_direction

        # Compute the heading vector
        heading = state[0] - state[1]
        heading_norm = np.linalg.norm(heading)
        heading_unit = heading / heading_norm
        displacement = heading_unit * (v * self.timestep)

        # Calculate the rotation matrix
        rotation_matrix = np.array([[np.cos(omega * self.timestep), -np.sin(omega * self.timestep)],
                                    [np.sin(omega * self.timestep), np.cos(omega * self.timestep)]])

        # Rotate the heading vector
        rotated_heading = rotation_matrix @ heading_unit

        # Calculate the new front and rear points
        next_front_point = state[0] + rotated_heading * heading_norm + displacement
        next_rear_point = state[1] - rotated_heading * heading_norm  + displacement

        # Combine front and rear points into the next_state matrix
        next_state = np.column_stack((next_front_point, next_rear_point)).T

            

        # check if the car is out of bounds
        if (np.any(next_state <= self.observation_space.low) or
                np.any(next_state >= self.observation_space.high)):
            done = False
            reward = 0
            info = {'is_legal': False}
            state = next_state

            if update_env:
                self.update_environment(next_state)
            return next_state, reward, done, info

        # calculate distance to goal and obstacle
        car_front = next_state[0]
        

       # dist_obstacle = np.linalg.norm(car_front - self.obstacle_position) - self.obstacle_radius

        dist_obstacle_front = np.linalg.norm(next_state[0] - self.obstacle_position) - self.obstacle_radius
        dist_obstacle_rear = np.linalg.norm(next_state[1] - self.obstacle_position) - self.obstacle_radius

        obstacle_collision = dist_obstacle_front < 0 or dist_obstacle_rear < 0


        if obstacle_collision:
            done = False
            reward = -1
            info = {'is_legal': False}
            next_state = state

            if update_env:
                self.update_environment(next_state)
            return next_state, reward, done, info



        #dist_goal = np.linalg.norm(car_front - self.goal_position) - self.min_distance_to_goal
        dist_goal_front = np.linalg.norm(next_state[0] - self.goal_position) - self.min_distance_to_goal
        dist_goal_rear = np.linalg.norm(next_state[1] - self.goal_position) - self.min_distance_to_goal

        goal_reached = dist_goal_front <= 0 or dist_goal_rear <= 0


        if goal_reached:
            state = next_state
            done = True
            reward = self.reward
            info = {'is_legal': True}

            if update_env:
                self.update_environment(next_state)
            return next_state, reward, done, info

        else:
            state = next_state
            reward = 0
            done = False
            info = {'is_legal': True}

            if update_env:
                self.update_environment(next_state)
            return next_state, reward, done, info



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

    def set(self, front_x, front_y, rear_x, rear_y):
        """
        Set the environment state with the given front and rear points of the car
        """
        self.state = np.array([[front_x, front_y], [rear_x, rear_y]], dtype=self.observation_space.dtype)

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

        # draw goal
        goal = plt.Circle((self.goal_position[0], self.goal_position[1]), self.min_distance_to_goal, color='g', fill=False)

        # draw obstacle
        obstacle = plt.Circle((self.obstacle_position[0], self.obstacle_position[1]), self.obstacle_radius, color='r', fill=False)
        plt.gca().add_artist(obstacle)
        plt.gca().add_artist(goal)

        # draw car (line connecting front and rear points)
        plt.plot(self.state[0, :], self.state[1, :], 'b-', linewidth=2)

        # draw heading arrow
        car_center = np.mean(self.state, axis=1)
        heading = self.state[:, 0] - self.state[:, 1]
        arrow_len = self.v_max
        arrow_dx = arrow_len * heading[0]
        arrow_dy = arrow_len * heading[1]

        plt.quiver(car_center[0], car_center[1], arrow_dx, arrow_dy, angles='xy', scale_units='xy', scale=1, width=0.01)
        plt.jet()

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






        

