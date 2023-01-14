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
        self.omega_max = 1 # maximum angular velocity
        
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
            return self.car_position, 0, True, {}
        # calculate distance to goal and obstacle
        dist_goal = np.linalg.norm(self.car_position - self.goal_position)
        dist_obstacle = np.linalg.norm(self.car_position - self.obstacle_position) - self.obstacle_radius
        if self.car_orientation < self.observation_space.low[2] or self.car_orientation > self.observation_space.high[2]:
            return self.car_position, 0, True, {}
        # calculate reward
        reward = -dist_goal - dist_obstacle

        # check if done
        done = False
        if dist_goal < self.min_distance_to_goal or dist_obstacle < self.min_distance_to_obstacle:
            done = True

        return self.car_position, reward, done, {}
            
    def reset(self):
        """
        Reset the environment and return the initial state
        """
        self.car_position = np.random.randn(2)
        self.car_orientation = np.random.rand() * 2 * np.pi
        self.goal_position = np.random.randn(2)
        self.obstacle_position = np.random.randn(2)
        return self.car_position
    
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



import numpy as np

class TileCoder:
    def __init__(self, ntiles, ntilings, state_bounds):
        """
        Initialize the tile coder

        Parameters
        ----------
        ntiles : list of int
            Number of tiles for each dimension of the state space
        ntilings : int
            Number of tilings to use
        state_bounds : numpy array of shape (state_dim, 2)
            lower and upper bounds of each dimension of the state space
        """
        self.ntiles = ntiles
        self.ntilings = ntilings
        self.state_bounds = state_bounds
        self.state_dim = state_bounds.shape[0]
        self.offset = np.random.rand(self.state_dim) # random offset for each tiling
        self.width = (self.state_bounds[:, 1] - self.state_bounds[:, 0]) / self.ntiles # width of each tile

    def get_tile_indices(self, state):
        """
        Get the tile indices for a given state

        Parameters
        ----------
        state : numpy array of shape (state_dim,)
            State for which to get the tile indices

        Returns
        -------
        tile_indices : list of int
            List of tile indices for the given state
        """
        tile_indices = []
        for tiling in range(self.ntilings):
            offset = self.offset[:, None] * tiling
            indices = np.floor((state - offset) / self.width).astype(int)
            # clamp indices to range [0, ntiles - 1]
            indices = np.maximum(indices, 0)
            indices = np.minimum(indices, self.ntiles - 1)
            # convert indices to a single integer
            index = np.sum(indices * np.prod(self.ntiles[:len(indices) - 1]) * self.ntilings + tiling)
            tile_indices.append(index)
        return tile_indices



# env = DubinsCarEnv()
# state = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     state, reward, done, _ = env.step(action)
#     env.render()

import gym

# Create the environment
env = DubinsCarEnv()

# Define the number of tiles, tilings, and state bounds
ntiles = [10, 10, 8]
ntilings = 8
state_bounds = np.array([[env.observation_space.low[0], env.observation_space.high[0]],
                         [env.observation_space.low[1], env.observation_space.high[1]],
                         [env.observation_space.low[2], env.observation_space.high[2]]])

# Create the tile coder
tc = TileCoder(ntiles, ntilings, state_bounds)

# Get a sample state from the environment
state = env.reset()

# Discretize the state using the tile coder
tile_indices = tc.get_tile_indices(state)

print("Tile indices:", tile_indices)

