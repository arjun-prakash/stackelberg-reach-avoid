import gym
import numpy as np

class DubinsCarEnv(gym.Env):
    def __init__(self):
        # Define the state space and action space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.action_space = gym.spaces.Discrete(3)

        # Define the car's parameters
        self.v = 1  # constant velocity
        self.steering_angle = np.pi / 3  # constant steering angle
        self.dt = 0.1  # time step
        
        # Define the goal and obstacle
        self.goal = np.array([5, 5])
        self.obstacle = np.array([[2, 2], [3, 3]])  # obstacle is a rectangle with corners (2, 2) and (3, 3)

        # Initialize the state
        self.state = np.array([0, 0, 0])  # x, y, theta

    def step(self, action):
        """
        Advance the car by one time step.
        """
        # Calculate the new state based on the action and the car's parameters
        if action == 0:  # left
            self.state[2] += self.steering_angle * self.dt
        elif action == 1:  # straight
            pass  # no change to theta
        elif action == 2:  # right
            self.state[2] -= self.steering_angle * self.dt

        self.state[0] += self.v * np.cos(self.state[2]) * self.dt
        self.state[1] += self.v * np.sin(self.state[2]) * self.dt

        # Check for collision with the obstacle
        if (self.state[0] > self.obstacle[0][0] and self.state[0] < self.obstacle[1][0] and
                self.state[1] > self.obstacle[0][1] and self.state[1] < self.obstacle[1][1]):
            done = True
            reward = -1000
        else:
            done = False
            reward = -1

        # Check for reaching the goal
        if np.linalg.norm(self.state[:2] - self.goal) < 0.1:
            done = True
            reward = 0

        return self.state, reward, done, {}

    def reset(self):
        """
        Reset the car's position and orientation to the initial state.
        """
        self.state = np.array([0, 0, 0])
        return self.state

    def render(self, mode='human'):
        """
        Render the environment.
        """
        pass  # Not implemented
