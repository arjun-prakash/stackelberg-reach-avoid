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

