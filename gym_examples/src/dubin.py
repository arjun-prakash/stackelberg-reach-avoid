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



class TileCoder:
    def __init__(self, num_tiles, x_range, y_range, num_tilings, offset=None):
        """
        Initialize the tile coder
        :param num_tiles: number of tiles for each dimension of the state space
        :param x_range: range of the x dimension of the state space
        :param y_range: range of the y dimension of the state space
        :param num_tilings: number of tilings to use
        :param offset: offset for each tiling
        """
        self.num_tiles = num_tiles
        self.x_range = x_range
        self.y_range = y_range
        self.num_tilings = num_tilings
        self.tiles = np.zeros((num_tiles, num_tiles, num_tilings))
        self.x_tile_size = (x_range[1] - x_range[0]) / num_tiles
        self.y_tile_size = (y_range[1] - y_range[0]) / num_tiles
        self.offsets = offset or np.random.rand(num_tilings, 2)

    def discretize(self, state):
        """
        Discretize the state into a tile
        :param state: continuous state [x, y, theta]
        :return: discrete state [i, j, theta]
        """
        i = np.zeros((self.num_tilings,))
        j = np.zeros((self.num_tilings,))
        for tiling_idx in range(self.num_tilings):
            i[tiling_idx] = np.floor((state[0] + self.offsets[tiling_idx][0]) / self.x_tile_size)
            j[tiling_idx] = np.floor((state[1] + self.offsets[tiling_idx][1]) / self.y_tile_size)
        return i, j, state[2]

    def activate(self, state):
        """
        Activate the tiles that correspond to the state
        :param state: continuous state [x, y, theta]
        :return: binary array of shape (num_tiles, num_tiles, num_tilings)
        """
        i, j, _ = self.discretize(state)
        i = np.floor(np.clip(i, 0, self.num_tiles - 1)).astype(int)
        j = np.floor(np.clip(j, 0, self.num_tiles - 1)).astype(int)
        for tiling in range(self.num_tilings):
            self.tiles[i, j, tiling] = 1
        return self.tiles


    def reset(self):
        """
        Reset the tile coder
        """
        self.tiles = np.zeros((self.num_tiles, self.num_tiles, self.num_tilings))



    def render(self, state, goal_pos, obstacle_pos):
        """
        Render the trajectory of the Dubins car on the discretized state space
        :param states: sequence of continuous states of the Dubins car
        :param goal_pos: continuous goal position [x, y]
        :param obstacle_pos: continuous obstacle position [x, y]
        """
        import matplotlib.pyplot as plt
        #plt.imshow(np.zeros((self.num_tiles, self.num_tiles)), cmap='gray', origin='lower')
        x, y, _ = self.discretize(state)
        plt.scatter(x, y, c='r', s=50)
        g_x, g_y, _ = self.discretize(np.append(goal_pos, [0]))
        o_x, o_y, _ = self.discretize(np.append(obstacle_pos, [0]))
        plt.scatter(g_x, g_y, c='g', s=50, marker='x')
        plt.scatter(o_x, o_y, c='b', s=50, marker='x')

        plt.savefig("disc.png")
   










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
#     print(state)
# import gym

# Create the environment
# Create the environment
# Create the environment


import numpy as np

env = DubinsCarEnv()

# Create an instance of the TileCoder
num_tiles = 50
x_range = env.observation_space.low[0], env.observation_space.high[0]
y_range = env.observation_space.low[1], env.observation_space.high[1]
num_tilings = 1
tile_coder = TileCoder(num_tiles, x_range, y_range, num_tilings)

# Run the agent for some episodes
num_episodes = 3
for episode in range(num_episodes):
    # Reset the environment and get the initial state
    state = env.reset()
    states = [state]
    goal_pos = env.goal_position
    obstacle_pos = env.obstacle_position
    print('l')
    done = False
    while not done:
        # Choose a random action
        action = env.action_space.sample()

        # Take the action and get the next state, reward, done flag, and info
        state, reward, done, _ = env.step(action)
        tile_coder.activate(state)

        states.append(state)

    # Render the trajectory of the Dubins car on the discretized state space
        tile_coder.render(state, goal_pos, obstacle_pos)
