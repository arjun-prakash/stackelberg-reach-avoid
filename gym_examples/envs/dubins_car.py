import gym
from gym import spaces
import numpy as np


class DubinsCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([-5, -5, -np.pi]), high=np.array([5, 5, np.pi]), dtype=np.float32)        
        self.goal_position = np.array([0,0]) # position of the goal
        self.obstacle_position = np.array([0,0]) # position of the obstacle
        self.obstacle_radius = 0.1 # radius of the obstacle
        self.car_position = np.array([0,0]) # position of the car
        self.car_orientation = 0 # orientation of the car in radians
        self.min_distance_to_goal = 0.1 # minimum distance to goal to consider the task as done
        self.min_distance_to_obstacle = 0.1 # minimum distance to obstacle to consider the task as done
        self.timestep = 0.1 # timestep in seconds
        self.v_max = 1 # maximum speed
        self.omega_max = .524  # maximum angular velocity (radians)
        self.images = []
        
    def step(self, action):
        """
        Perform the action and return the next state, reward and done flag
        """
        v = self.v_max # speed of the car
        omega = self.omega_max # angular velocity of the car
        if action == 0: # turn left
            omega = -omega
        elif action == 2: # turn right
            omega = omega
        else: # action 1 : straight
            omega = 0

        # update car position and orientation
        self.car_orientation += omega * self.timestep
        self.car_position[0] += v * np.cos(self.car_orientation) * self.timestep
        self.car_position[1] += v * np.sin(self.car_orientation) * self.timestep

        # check if the car is out of bounds
        if self.car_position[0] < self.observation_space.low[0] or self.car_position[0] > self.observation_space.high[0] or self.car_position[1] < self.observation_space.low[1] or self.car_position[1] > self.observation_space.high[1]:
            print('out of bounds')
            return np.append(self.car_position, [self.car_orientation]), 0, True, {}
        # calculate distance to goal and obstacle
        dist_goal = np.linalg.norm(self.car_position - self.goal_position)
        dist_obstacle = np.linalg.norm(self.car_position - self.obstacle_position) - self.obstacle_radius
        if dist_obstacle < 0:
            print('wat')
            return np.append(self.car_position, [self.car_orientation]), 0, True, {}
        # calculate reward
        reward = -dist_goal - dist_obstacle

        # check if done
        done = False
        if dist_goal < self.min_distance_to_goal or dist_obstacle < self.min_distance_to_obstacle:
            done = True
            print('hit something')

        return np.append(self.car_position, [self.car_orientation]), reward, done, {}


    def state_action_step(self, state, action):
        """
        Perform the action and return the next state, reward and done flag
        """
        v = self.v_max # speed of the car
        omega = self.omega_max # angular velocity of the car
        if action == 0: # turn left
            omega = -omega
        elif action == 2: # turn right
            omega = omega
        else: # action 1 : straight
            omega = 0

        car_orientation = state[2]
        car_position = np.array(state[:2])
        # update car position and orientation
        car_orientation += omega * self.timestep
        car_position[0] += v * np.cos(car_orientation) * self.timestep
        car_position[1] += v * np.sin(car_orientation) * self.timestep

        # check if the car is out of bounds
        if car_position[0] < self.observation_space.low[0] or car_position[0] > self.observation_space.high[0] or car_position[1] < self.observation_space.low[1] or car_position[1] > self.observation_space.high[1]:
            print("out of bounds")
            return np.append(car_position, [car_orientation]), 0, True, {}
        # calculate distance to goal and obstacle
        dist_goal = np.linalg.norm(car_position - self.goal_position)
        dist_obstacle = np.linalg.norm(car_position - self.obstacle_position) - self.obstacle_radius
        if dist_obstacle < 0:
            print('wat')
            return np.append(car_position, [car_orientation]), 0, True, {}
        # calculate reward
        reward = -dist_goal - dist_obstacle

        # check if done
        done = False
        if dist_goal < self.min_distance_to_goal or dist_obstacle < self.min_distance_to_obstacle:
            print('hit something')
            done = True

        return np.append(car_position, [car_orientation]), reward, done, {}


    def sample(self, state, action, gamma):
        state_, reward, done, _ = self.state_action_step(state, action)

        next_rewards = []
        for a in range(self.action_space.n):
            _, reward_, _, _ = self.state_action_step(state, a)
            next_rewards.append(reward_)
        expected_next_reward = np.mean(next_rewards)

        return reward + gamma*expected_next_reward



            
            
    def reset(self):
        """
        Reset the environment and return the initial state
        """
        self.car_position = np.random.randn(2)
        self.car_orientation = np.random.rand() * 2 * np.pi
        self.goal_position = np.random.randn(2)
        self.obstacle_position = np.random.randn(2)
        return np.append(self.car_position, [self.car_orientation])
    
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

        # draw car
        car_x = [self.car_position[0], self.car_position[0] + np.cos(self.car_orientation)]
        car_y = [self.car_position[1], self.car_position[1] + np.sin(self.car_orientation)]
        plt.plot(car_x, car_y, 'k-')

        # draw goal
        goal = plt.Circle((self.goal_position[0], self.goal_position[1]), self.min_distance_to_goal, color='g', fill=False)

        # draw obstacle
        obstacle = plt.Circle((self.obstacle_position[0], self.obstacle_position[1]), self.obstacle_radius, color='r', fill=False)
        plt.gca().add_artist(obstacle)
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

