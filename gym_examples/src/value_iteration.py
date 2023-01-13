import gym
import numpy as np

class ValueIteration:
    def __init__(self, env, gamma=0.99, theta=1e-10):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.observation_space.n)
        self.policy = np.zeros([env.observation_space.n, env.action_space.n])

    def value_iteration(self):
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.V[s]

                q_values = [self.q(s, a) for a in range(self.env.action_space.n)]
                self.V[s] = max(q_values)

                delta = max(delta, abs(v - self.V[s]))

            if delta < self.theta:
                break

        for s in range(self.env.observation_space.n):
            q_values = [self.q(s, a) for a in range(self.env.action_space.n)]
            self.policy[s] = np.eye(self.env.action_space.n)[np.argmax(q_values)]


    def q(self, s, a):
        return sum(p * (r + self.gamma * self.V[s_]) for p, s_, r, _ in self.env.P[s][a])

    def run(self):
        self.value_iteration()
        return self.policy



class ValueIteration2:
    def __init__(self, env, gamma=0.99, theta=1e-10):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.nS)
        self.policy = np.zeros([env.nS, env.nA])

    def value_iteration(self):
        while True:
            delta = 0
            for s in range(self.env.nS):
                v = self.V[s]

                q_values = [self.q(s, a) for a in range(self.env.nA)]
                self.V[s] = max(q_values)

                delta = max(delta, abs(v - self.V[s]))

            if delta < self.theta:
                break

        for s in range(self.env.nS):
            q_values = [self.q(s, a) for a in range(self.env.nA)]
            self.policy[s] = np.eye(self.env.nA)[np.argmax(q_values)]

    def q(self, s, a):
        return sum(p * (r + self.gamma * self.V[s_]) for p, s_, r, _ in self.env.P[s][a])

    def run(self):
        self.value_iteration()
        return self.policy


class TileCodedValueIteration:
    def __init__(self, env, tc, gamma=0.99, theta=1e-10):
        """
        Initialize the value iteration algorithm.
        
        Parameters:
            env (gym.Env): The environment.
            tc (TileCoder): The tile coder.
            gamma (float): The discount factor.
            theta (float): The threshold for convergence.
        """
        self.env = env
        self.tc = tc
        self.gamma = gamma
        self.theta = theta
        # Number of states is the product of the number of tiles in each dimension
        self.n_states = np.prod(self.tc.n_tiles)
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros([self.n_states, self.env.action_space.n])

    def value_iteration(self):
        """
        Perform the value iteration algorithm.
        """
        while True:
            delta = 0
            for s in range(self.n_states):
                v = self.V[s]
                # Convert the state index to tile indices
                state_tiles = np.unravel_index(s, self.tc.n_tiles)
                q_values = [self.q(state_tiles, a) for a in range(self.env.action_space.n)]
                self.V[s] = max(q_values)
                delta = max(delta, abs(v - self.V[s]))

            if delta < self.theta:
                break

        for s in range(self.n_states):
            state_tiles = np.unravel_index(s, self.tc.n_tiles)
            q_values = [self.q(state_tiles, a) for a in range(self.env.action_space.n)]
            self.policy[s] = np.eye(self.env.action_space.n)[np.argmax(q_values)]

    def q(self, state_tiles, a):
        """
        Compute the Q-value for a given state and action.
        
        Parameters:
            state_tiles (tuple): The tile indices of the state.
            a (int): The action.
            
        Returns:
            float: The Q-value.
        """
        return sum(p * (r + self.gamma * self.V[s_]) for p, s_, r, _ in self.env.P[state_tiles][a
