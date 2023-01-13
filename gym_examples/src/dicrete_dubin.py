import gym
import numpy as np
from gym import spaces

class DubinsCarEnv(gym.Env):
    """
    A 2D Dubins car environment for OpenAI Gym.
    The car has a fixed turning radius and a constant speed. The action space is either left, straight or right.
    The car must reach a goal and avoid an obstacle.
    """

    def __init__(self, speed=1, turning_radius=1, goal_state=np.array([5,5,0]), obstacle_state=np.array([3,3,0]), num_tilings=8, tile_resolution=0.1, num_bins=8):
        """
        Initialize the DubinsCar environment.
        :param speed: The speed of the car (default: 1)
        :param turning_radius: The turning radius of the car (default: 1)
        :param goal_state: The goal state of the car (default: [5,5,0])
        :param obstacle_state: The obstacle state of the car (default: [3,3,0])
        :param num_tilings: The number of tilings for tile coding (default: 8)
        :param tile_resolution: The resolution of the tiles for tile coding (default: 0.1)
        :param num_bins: The number of bins for discretizing the heading angle (default: 8)
        """
        self.action_space = spaces.Discrete(3)  # left, straight, right
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))

        self.speed = speed
        self.turning_radius = turning_radius
        self.goal_state = goal_state
        self.obstacle_state = obstacle_state

        self.num_tilings = num_tilings
        self.tile_resolution = tile_resolution
        self.num_bins = num_bins
        self.num_states = (2 * self.num_bins + 1) ** self.num_tilings

        # initialize the value table
        self.value_table = np.zeros(self.num_states)

    def _get_state(self, position, heading, tiling_index):
        """
        Get the state code using tile coding.
        :param position: The position of the car
        :param heading: The heading angle of the car
        :param tiling_index: The index of the current tiling
        :return: The state code
        """
        offset = np.array([tiling_index/self.num_tilings, tiling_index/self.num_tilings])
        x_bin = int((position[0] + offset[0]) / self.tile_resolution)
        y_bin = int((position[1] + offset[1]) / self.tile_resolution)
        h_bin = int(heading / (2 * np.pi) * self.num_bins)
        state_code = x_bin * (self.num_bins ** 2) + y_bin * self.num_bins + h_bin
        return state_code

    def step(self, action):
        """
        Step the car according to the given action and update the state.
        :param action: The action to take (left, straight, or right)
        :return: A tuple (state, reward, done, info)
        """
        assert self.action_space.contains(action)

        # update the state based on the action
        if action == 0:  # left
            self.state[2] += np.pi/3
        elif action == 1:  # straight
            self.state[0] += self.speed * np.cos(self.state[2])
            self.state[1] += self.speed * np.sin(self.state[2])
        else: #right        
            self.state[2] -= np.pi/3

        # check for collisions
        if np.linalg.norm(self.state[:2]-self.obstacle_state[:2]) < 0.5:
            return self.state, -1, True, {}

        # check if goal is reached
        if np.linalg.norm(self.state[:2]-self.goal_state[:2]) < 0.5:
            return self.state, 1, True, {}

        state_code = 0
        for i in range(self.num_tilings):
            state_code += self._get_state(self.state[:2], self.state[2], i)
        return self.state, 0, False, {'state_code': state_code}

    def reset(self):
        """
        Reset the car to the initial state.
        :return: The initial state of the car.
        """
        self.state = np.array([0, 0, np.pi/2])
        state_code = self._get_state(self.state[:2], self.state[2], 0)
        return self.state



class ValueIteration:
    def __init__(self, env, discount_factor=0.99, max_iterations=1000, threshold=1e-8):
        self.env = env
        self.discount_factor = discount_factor
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.value_table = np.zeros(self.env.num_states)
        self.q_table = np.zeros((self.env.num_states, self.env.action_space.n))
    
    def value_iteration(self):
        for i in range(self.max_iterations):
            delta = 0
            for state_code in range(self.env.num_states):
                v = self.value_table[state_code]
                for action in range(self.env.action_space.n):
                    _, reward, _, _ = self.env.step(action)
                    next_state_code = self.env._get_state(self.env.state[:2], self.env.state[2], 0)
                    self.q_table[state_code][action] = reward + self.discount_factor * self.value_table[next_state_code]
                self.value_table[state_code] = max(self.q_table[state_code])
                delta = max(delta, abs(v - self.value_table[state_code]))
            if delta < self.threshold:
                break
        return self.value_table
    
    def run(self):
        return self.value_iteration()
    
    def q(self):
        return self.q_table


vi = ValueIteration(DubinsCarEnv(), discount_factor=0.99, max_iterations=1000, threshold=1e-8)
values = vi.run()
q_table = vi.q()

policy = np.zeros(env.num_states)
for state_code in range(env.num_states):
    policy[state_code] = np.argmax(q_table[state_code])

done = False
while not done:
    state = env.reset()
    action = policy[state_code]
    state, reward, done, _ = env.step(action)
    env.render()
