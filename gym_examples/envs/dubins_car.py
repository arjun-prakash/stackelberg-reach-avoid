import gym
from gym import spaces
import numpy as np


class DubinsCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([-5, -5, -np.pi]), high=np.array([5, 5, np.pi]), dtype=np.float32)        
        self.goal_position = np.array([0,0]) # position of the goal
        self.obstacle_position = np.array([2,2]) # position of the obstacle
        self.obstacle_radius = 0.1 # radius of the obstacle
        self.car_position = np.array([0,0]) # position of the car
        self.car_orientation = 0 # orientation of the car in radians
        self.min_distance_to_goal = 1 # minimum distance to goal to consider the task as done
        self.min_distance_to_obstacle = 0.1 # minimum distance to obstacle to consider the task as done
        self.timestep = 0.5 # timestep in seconds
        self.v_max = 0.5 # maximum speed
        self.omega_max = 0.524  # maximum angular velocity (radians)
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
            return np.append(self.car_position, [self.car_orientation]), -10, True, {}
        # calculate distance to goal and obstacle
        dist_goal = np.linalg.norm(self.car_position - self.goal_position)
        dist_obstacle = np.linalg.norm(self.car_position - self.obstacle_position) - self.obstacle_radius
        if dist_obstacle < 0:
            print('wat')
            return np.append(self.car_position, [self.car_orientation]), -10, True, {}
        # calculate reward
        reward = -dist_goal 

        # check if done
        done = False
        if dist_goal < self.min_distance_to_goal:
            done = True
            print('hit goal')
            reward=10

        #print('other', reward)
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
            #print("out of bounds")
            return np.append(car_position, [car_orientation]), -10, True, {}
        # calculate distance to goal and obstacle
        dist_goal = np.linalg.norm(car_position - self.goal_position)
        dist_obstacle = np.linalg.norm(car_position - self.obstacle_position) - self.obstacle_radius
        if dist_obstacle < 0:
            #print('wat')
            return np.append(car_position, [car_orientation]), -10, True, {}
        # calculate reward
        reward = -dist_goal

        # check if done
        done = False
        if dist_goal < self.min_distance_to_goal:
            done = True
            reward = 10
            #print('goal', reward)

        #print('other', reward)
        return np.append(car_position, [car_orientation]), reward, done, {}


    def sample(self, state, action, gamma):
        state_, reward, done, _ = self.state_action_step(state, action)

        next_rewards = []
        for a in range(self.action_space.n):
            _, reward_, _, _ = self.state_action_step(state_, a)
            next_rewards.append(reward_)
        expected_next_reward = np.max(next_rewards)

        return reward + gamma*expected_next_reward


    def sample_value_iter(self,X_batch, forward, params, gamma):
        y_hat = []
        for state in X_batch:
            next_rewards = []
            for action in range(self.action_space.n):

                state_, reward, done, _ = self.state_action_step(state, action)
                discounted_reward = reward + gamma*forward(X=state, params=params)

                next_rewards.append(discounted_reward)
            best_next_reward = np.max(next_rewards)
            y_hat.append(best_next_reward)

        return np.array(y_hat)




            
            
    def reset(self):
        """
        Reset the environment and return the initial state
        """
        self.car_position = self.observation_space.sample()[:2]
        self.car_orientation = self.observation_space.sample()[2]
        self.goal_position = np.array([0,0]) 
        self.obstacle_position = np.array([2,2]) 
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



class TwoPlayerDubinsCarEnv(DubinsCarEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()


        self.players = ['defender', 'attacker']

        self.action_space = {'attacker':spaces.Discrete(3), 'defender':spaces.Discrete(3)}

        self.observation_space= {'attacker':spaces.Box(low=np.array([-5, -5, -np.pi]), high=np.array([5, 5, np.pi]), dtype=np.float32), 
                                'defender':spaces.Box(low=np.array([-5, -5, -np.pi]), high=np.array([5, 5, np.pi]), dtype=np.float32)}



        self.car_position = {'attacker': np.array([0,0,0]), 'defender':np.array([0,0,0])}

        self.goal_position = np.array([0,0]) # position of the goal
        self.capture_radius = 0.5 # radius of the obstacle



        self.min_distance_to_goal = 1 # minimum distance to goal to consider the task as done
        self.min_distance_to_obstacle = 0.1 # minimum distance to obstacle to consider the task as done

        self.timestep = 0.2 # timestep in seconds
        self.v_max = 0.2 # maximum speed
        self.omega_max = .524  # maximum angular velocity (radians)
        self.images = []
        

    def reset(self):
        """
        Reset the environment and return the initial state
        """
        self.car_position['attacker'] = self.observation_space['attacker'].sample()
        self.car_position['defender'] = self.observation_space['defender'].sample()
        #self.car_position['defender'] = np.array([2,2,2])


        self.goal_position = np.array([0,0]) 
        print('reset',self.car_position)
        return self.car_position
    


    def step(self, action, player):
        """
        Perform the action and return the next state, reward and done flag
        action: dict{attacker:int, defender:int}
        """


            
        v = self.v_max # speed of the car
        omega = self.omega_max # angular velocity of the car
        if action == 0: # turn left
            omega = - omega
        elif action == 2: # turn right
            omega = omega
        else: # action 1 : straight
            omega = 0

        # update car position and orientation
        self.car_position[player][2] += omega * self.timestep
        self.car_position[player][0] += v * np.cos(self.car_position[player][2]) * self.timestep
        self.car_position[player][1] += v * np.sin(self.car_position[player][2]) * self.timestep

        # # check if the car is out of bounds
        # if self.car_position[player][0] < self.observation_space[player].low[0] or self.car_position[player][0] > self.observation_space[player].high[0] or self.car_position[player][1] < self.observation_space[player].low[1] or self.car_position[player][1] > self.observation_space[player].high[1]:
        #     print('out of bounds')
        #     return self.car_position, -10, True, {}

        # check if the car is out of bounds
        if self.car_position['attacker'][0] < self.observation_space['attacker'].low[0] or self.car_position['attacker'][0] > self.observation_space['attacker'].high[0] or self.car_position['attacker'][1] < self.observation_space['attacker'].low[1] or self.car_position['attacker'][1] > self.observation_space[player].high[1]:
            print('out of bounds')
            return self.car_position, -10, True, {}
        
        
       
        # calculate distance to goal and obstacle
        dist_goal = np.linalg.norm(self.car_position['attacker'][:2] - self.goal_position)

        dist_capture = np.linalg.norm(self.car_position['attacker'][:2] - self.car_position['defender'][:2]) - self.capture_radius
        if dist_capture < 0:
            print('captured')
            return self.car_position, -10, True, {}
        # calculate reward
        reward = -dist_goal

        # check if done
        done = False
        if dist_goal < self.min_distance_to_goal:
            done = True
            print('gaol!')

        return self.car_position, reward, done, {}


    def state_action_step(self,state, action, player):
        """
        Perform the action and return the next state, reward and done flag
        action: dict{attacker:int, defender:int}
        """


            
        v = self.v_max # speed of the car
        omega = self.omega_max # angular velocity of the car
        if action == 0: # turn left
            omega = - omega
        elif action == 2: # turn right
            omega = omega
        else: # action 1 : straight
            omega = 0

        car_position = state        
        # update car position and orientation
        car_position[player][2] += omega * self.timestep
        car_position[player][0] += v * np.cos(car_position[player][2]) * self.timestep
        car_position[player][1] += v * np.sin(car_position[player][2]) * self.timestep

        # check if the car is out of bounds
        if car_position['attacker'][0] < self.observation_space['attacker'].low[0] or car_position['attacker'][0] > self.observation_space[player].high[0] or car_position['attacker'][1] < self.observation_space[player].low[1] or car_position['attacker'][1] > self.observation_space[player].high[1]:
            print('out of bounds', player)
            print(car_position[player])
            return car_position, -10, True, {}
        
        
       
        # calculate distance to goal and obstacle
        dist_goal = np.linalg.norm(car_position['attacker'][:2] - self.goal_position)

        dist_capture = np.linalg.norm(car_position['attacker'][:2] - car_position['defender'][:2]) - self.obstacle_radius
        if dist_capture < 0:
            print('captured')
            return car_position, -10, True, {}
        # calculate reward
        reward =  -dist_goal

        # check if done
        done = False
        if dist_goal < self.min_distance_to_goal:
            done = True
            reward = 10
            print('goal!')

        return car_position, reward, done, {}



    def sample(self, state, action,player,gamma):
        state_, reward, done, _ = self.state_action_step(state, action, player)

        next_rewards = []

        for player in self.players:
            for a in range(self.action_space[player].n):
                _, reward_, _, _ = self.state_action_step(state, a, player)
                next_rewards.append(reward_)
            expected_next_reward = np.mean(np.array(next_rewards), axis=0)

        return reward + gamma*expected_next_reward



            


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

    

        attacker = plt.Circle((self.car_position['attacker'][0], self.car_position['attacker'][1]), 0.1, color='b', fill=True)
        defender = plt.Circle((self.car_position['defender'][0], self.car_position['defender'][1]), self.capture_radius, color='r', fill=True)


        # draw goal
        goal = plt.Circle((self.goal_position[0], self.goal_position[1]), self.min_distance_to_goal, color='g', fill=False)

        # draw obstacle
        plt.gca().add_artist(attacker)
        plt.gca().add_artist(defender)
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

        imageio.mimsave('two_player_animation.gif', self.images, fps=30)

