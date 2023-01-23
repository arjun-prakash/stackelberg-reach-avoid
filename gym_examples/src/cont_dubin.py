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
            return np.append(self.car_position, [self.car_orientation]), 0, True, {}
        # calculate distance to goal and obstacle
        dist_goal = np.linalg.norm(self.car_position - self.goal_position)
        dist_obstacle = np.linalg.norm(self.car_position - self.obstacle_position) - self.obstacle_radius
        if self.car_orientation < self.observation_space.low[2] or self.car_orientation > self.observation_space.high[2]:
            return np.append(self.car_position, [self.car_orientation]), 0, True, {}
        # calculate reward
        reward = -dist_goal - dist_obstacle

        # check if done
        done = False
        if dist_goal < self.min_distance_to_goal or dist_obstacle < self.min_distance_to_obstacle:
            done = True

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
            return np.append(car_position, [car_orientation]), 0, True, {}
        # calculate distance to goal and obstacle
        dist_goal = np.linalg.norm(car_position - self.goal_position)
        dist_obstacle = np.linalg.norm(car_position - self.obstacle_position) - self.obstacle_radius
        if car_orientation < self.observation_space.low[2] or car_orientation > self.observation_space.high[2]:
            return np.append(car_position, [car_orientation]), 0, True, {}
        # calculate reward
        reward = -dist_goal - dist_obstacle

        # check if done
        done = False
        if dist_goal < self.min_distance_to_goal or dist_obstacle < self.min_distance_to_obstacle:
            done = True

        return np.append(car_position, [car_orientation]), reward, done, {}
            
            
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
        matplotlib.use('Agg')

        plt.clf()
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])

        # draw car
        car_x = [self.car_position[0], self.car_position[0] + np.cos(self.car_orientation)]
        car_y = [self.car_position[1], self.car_position[1] + np.sin(self.car_orientation)]
        plt.plot(car_x, car_y, 'k-')

        # draw goal
        plt.plot(self.goal_position[0], self.goal_position[1], 'go')

        # draw obstacle
        obstacle = plt.Circle((self.obstacle_position[0], self.obstacle_position[1]), self.obstacle_radius, color='r', fill=False)
        plt.gca().add_artist(obstacle)
        print('saving')
        plt.savefig("mygraph.png")
       # plt.pause(0.1)

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




# env = DubinsCarEnv()
# state = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     state, reward, done, _ = env.step(action)
#     env.render()
#     print(state)
# import gym

# Create the environment
# Create the environment
# Create the environment


import numpy as np

env = DubinsCarEnv()

# Create an instance of the TileCod/watcher
num_tiles = 3
x_range = env.observation_space.low[0], env.observation_space.high[0]
y_range = env.observation_space.low[1], env.observation_space.high[1]
theta_range = env.observation_space.low[2], env.observation_space.high[2]

num_tilings = 2
tile_coder = TileCoder(num_tiles, x_range, y_range, theta_range, num_tilings)

# # Run the agent for some episodes
# num_episodes = 3
# for episode in range(num_episodes):
#     # Reset the environment and get the initial state
#     state = env.reset()
#     states = [state]
#     goal_pos = env.goal_position
#     obstacle_pos = env.obstacle_position
#     print('l')
#     done = False
#     while not done:
#         # Choose a random action
#         action = env.action_space.sample()

#         # Take the action and get the next state, reward, done flag, and info
#         state, reward, done, _ = env.step(action)
#         tile_coder.activate(state)

#         states.append(state)

#     # Render the trajectory of the Dubins car on the discretized state space
#     tile_coder.render(state, goal_pos, obstacle_pos)

from rich import print
vi = ValueIteration(env, tile_coder)
vi.solve()
p = vi.get_policy()
print(p)
print('done')