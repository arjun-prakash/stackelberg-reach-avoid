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
        self.omega_max = 1 # maximum angular velocity (radians)
        
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



class TileCoder:
    def __init__(self, ntiles, ntilings, state_bounds):
        """
        Initialize the tile coder

        Parameters
        ----------
        ntiles : list of int
            Number of tiles for each dimension of the state space
        ntilings : int
            Number of tilings
        state_bounds : numpy array of shape (state_dim, 2)
            Lower and upper bounds for each dimension of the state space
        """
        self.ntiles = ntiles
        self.ntilings = ntilings
        self.state_dim = len(ntiles)
        self.state_bounds = state_bounds
        self.width = (state_bounds[:, 1] - state_bounds[:, 0]) / ntiles
        self.offset = state_bounds[:, 0]

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
        state = np.append(state, state[-1] % (2 * np.pi))
        tile_indices = []
        for tiling in range(self.ntilings):
            indices = []
            for i in range(self.state_dim):
                index = np.floor((state[i] - self.offset[i]) / self.width[i]).astype(int)
                index = max(0, index)
                if i == self.state_dim-1:
                    index = min(index, self.ntiles[-1]-1)
                indices.append(index)
            index = np.sum(indices) + tiling * np.prod(self.ntiles)
            tile_indices.append(index)
        return tile_indices





class DiscreteDubinsEnv(DubinsCarEnv):
    def __init__(self, ntiles, ntilings, state_bounds):
        super().__init__()
        self.tc = TileCoder(ntiles, ntilings, state_bounds)
        self.ntiles = ntiles
        self.width = (state_bounds[:, 1] - state_bounds[:, 0]) / ntiles
        self.offset = state_bounds[:, 0]

    def state_to_obs(self, state):
        """
        Convert the state index to observation format

        Parameters
        ----------
        state : int
            The state index to convert

        Returns
        -------
        obs : numpy array of shape (3,)
            The observation in the format (x, y, theta)
        """
        x,y,theta = state%np.prod(self.ntiles[:2]), state%np.prod(self.ntiles[:1]), state%np.prod(self.ntiles[2])
        x = self.offset[0]+self.width[0]*x
        y = self.offset[1]+self.width[1]*y
        theta = self.offset[2]+self.width[2]*theta
        obs = np.array([x, y, theta])
        return obs
    
    def obs_to_state(self, obs):
        """
        Convert the observation format to the corresponding state index

        Parameters
        ----------
        obs : numpy array of shape (3,)
            The observation in the format (x, y, theta)

        Returns
        -------
        state : int
            The state index corresponding to the observation
        """
        x, y, theta = obs
        x_index = int((x - self.offset[0]) / self.width[0])
        y_index = int((y - self.offset[1]) / self.width[1])
        theta_index = int((theta - self.offset[2]) / self.width[2])
        state = x_index + y_index * self.ntiles[0] + theta_index * self.ntiles[0] * self.ntiles[1]
        return state




class ValueIteration:
    def __init__(self, env, tc, discount_factor=0.99, theta=1e-8):
        """
        Initialize the value iteration algorithm

        Parameters
        ----------
        env : gym.Env
            The environment to solve
        tc : TileCoder
            The tile coder to use for discretizing the state space
        discount_factor : float, optional
            The discount factor, by default 0.99
        theta : float, optional
            The stopping criterion, by default 1e-8
        """
        self.env = env
        self.tc = tc
        self.discount_factor = discount_factor
        self.theta = theta
        self.n_states = np.prod(tc.ntiles) * tc.ntilings
        self.values = np.zeros((self.n_states))
        self.policy = np.zeros((self.n_states))
        
    def solve(self):
        """
        Solve the environment using value iteration
        """
        while True:
            delta = 0
            for state in range(self.n_states):
                state_indices = self.tc.get_tile_indices(self.env.state_to_obs(state))
                v = self.values[state_indices]
                action_values = []
                for action in range(self.env.action_space.n):
                    next_state_indices, reward, done, _ = self.env.step(action)
                    action_values.append(reward + self.discount_factor * self.values[next_state_indices])
                new_v = np.max(action_values)
                delta = max(delta, np.abs(v - new_v))
                self.values[state_indices] = new_v
                self.policy[state_indices] = np.argmax(action_values)
            if delta < self.theta:
                break



# env = DubinsCarEnv()
# state = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     state, reward, done, _ = env.step(action)
#     env.render()

import gym

# Create the environment
# Create the environment
# Create the environment


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

# Create the value iteration object
vi = ValueIteration(env, tc)

# Solve the environment
vi.solve()

# Print the final values and policy
print("Final values:", vi.values)
print("Final policy:", vi.policy)
