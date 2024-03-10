import jax
import jax.numpy as jnp
from gym import spaces
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import io
import imageio
matplotlib.use('Agg')

class ContinuousTwoPlayerEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, size, max_steps, capture_radius, goal_position, goal_radius, timestep, v_max):
        self.players = ['defender', 'attacker']
        self.action_space = {
            'attacker': spaces.Box(low=-1, high=1, shape=(2,), dtype=jnp.float32),
            'defender': spaces.Box(low=-1, high=1, shape=(2,), dtype=jnp.float32)
        }

        self.size = size
        self.max_steps = max_steps
        # self.observation_space = {
        #     'attacker': spaces.Box(low=jnp.array([-self.size, -self.size]), high=jnp.array([self.size, self.size]), dtype=jnp.float32),
        #     'defender': spaces.Box(low=jnp.array([-self.size, -self.size]), high=jnp.array([self.size, self.size]), dtype=jnp.float32)
        # }

        self.capture_radius = capture_radius
        self.goal_position = jnp.array(goal_position)
        self.goal_radius = goal_radius
        self.timestep = timestep
        self.v_max = v_max
        self.step_count = 0
        self.state = None
        self.images = []
        self.positions = {'attacker': [], 'defender': []}


    # def reset(self, key):
    #     key_defender, key_attacker = jax.random.split(key)
    #     state = {
    #         'attacker': jax.random.uniform(key_attacker, minval=-self.size, maxval=self.size, shape=(2,)),
    #         'defender': jax.random.uniform(key_defender, minval=-self.size, maxval=self.size, shape=(2,))
    #     }
    #     nn_state = self.encode_helper(state)

    #     return key, state, nn_state
    
    def reset(self, key):
        # Use JAX random to set initial positions if needed
        key, subkey1, subkey2 = jax.random.split(key, 3)
        down = 3 * np.pi / 2
        up = np.pi / 2

        # Example of setting random positions; adapt as necessary for your environment
        defender_x = jax.random.uniform(subkey1, minval=0, maxval=0)
        defender_y = jax.random.uniform(subkey1, minval=-1, maxval=-1)

        attacker_x = jax.random.uniform(subkey2, minval=0, maxval=0)
        attacker_y = jax.random.uniform(subkey2, minval=2.9, maxval=2.9)

        # Set initial state using provided positions or randomized ones
        state = {
            'defender': jnp.array([defender_x, defender_y]),
            'attacker': jnp.array([attacker_x, attacker_y])
        }

        nn_state = self.encode_helper(state)

        return key, state, nn_state
    
    

    def _update_state(self, state, action, player):
        x, y = state[player]
        step_size = jax.lax.cond(player == 'attacker', lambda _: self.v_max, lambda _: self.v_max/8, None)

        #step_size = self.v_max * self.timestep
        new_x = x + action[0] * step_size
        new_y = y + action[1] * step_size
        new_x, new_y = self._restrict_movement(x, y, new_x, new_y)
        new_state = {**state, player: jnp.array([new_x, new_y])}
        new_nn_state = self.encode_helper(new_state)
        return new_state, new_nn_state

    def _restrict_movement(self, x, y, new_x, new_y):
        def within_radius():
            return new_x, new_y

        def outside_radius():
            theta = jnp.arctan2(new_y - y, new_x - x)
            return x + jnp.cos(theta) * self.v_max, y + jnp.sin(theta) * self.v_max

        dist = jnp.sqrt((new_x - x)**2 + (new_y - y)**2)
        return jax.lax.cond(dist <= self.v_max, within_radius, outside_radius)
    

    def _get_contraint(self, state):
            dist_capture = jnp.linalg.norm(state['attacker'][:2] - state['defender'][:2]) - self.capture_radius - self.v_max
            return -(dist_capture)






    # def _restrict_movement(self, x, y, new_x, new_y, state, player):
    #     def within_radius():
    #         return new_x, new_y

    #     def outside_radius(safe_step):
    #         theta = jnp.arctan2(new_y - y, new_x - x)
    #         return x + jnp.cos(theta) * safe_step, y + jnp.sin(theta) * safe_step

    #     dist = jnp.sqrt((new_x - x)**2 + (new_y - y)**2)

    #     # Corrected: Compute distance from attacker to defender and adjust for capture radius
    #     attacker_to_defender_dist = jnp.linalg.norm(state['attacker'][:2] - state['defender'][:2])
    #     dist_capture = attacker_to_defender_dist - self.capture_radius - 0.01

    #     # Determine if the player is the attacker and adjust the safe_step accordingly
    #     is_attacker = player == 'attacker'
    #     safe_step = jax.lax.cond(is_attacker,
    #                             lambda: jnp.minimum(self.v_max, dist_capture),
    #                             lambda: self.v_max)
        


    #     # Use safe_step to determine if movement adjustment is needed
    #     action = jax.lax.cond(dist <= self.v_max,
    #                         within_radius,
    #                         lambda: outside_radius(safe_step))
    #     return action
    

    # def _restrict_movement(self, x, y, new_x, new_y):
    #     def within_radius():
    #         return new_x, new_y
 
    #     def outside_radius(safe_step):
    #         theta = jnp.arctan2(new_y - y, new_x - x)
    #         return x + jnp.cos(theta) * safe_step, y + jnp.sin(theta) * safe_step

    #     dist = jnp.sqrt((new_x - x)**2 + (new_y - y)**2)

    #     # Corrected: Compute distance from attacker to defender and adjust for capture radius
    #     attacker_to_defender_dist = jnp.linalg.norm(state['attacker'][:2] - state['defender'][:2])
    #     dist_capture = attacker_to_defender_dist - self.capture_radius - 0.01

    #     # Determine if the player is the attacker and adjust the safe_step accordingly
    #     is_attacker = player == 'attacker'
    #     safe_step = jax.lax.cond(is_attacker,
    #                             lambda: jnp.minimum(self.v_max, dist_capture),
    #                             lambda: self.v_max)
    #     #safe_step = jax.lax.cond(safe_step != self.v_max, lambda:-self.v_max, lambda:self.v_max)
                
        

    #     # Use safe_step to determine if movement adjustment is needed
    #     action = jax.lax.cond(dist <= self.v_max,
    #                         within_radius,
    #                         lambda: outside_radius(safe_step))
    #     return action



  






   


    def _get_reward_done(self, state):
        dist_goal = jnp.linalg.norm(state['attacker'] - self.goal_position)
        done_win = dist_goal < self.goal_radius
        done_loss = jnp.linalg.norm(state['attacker'] - state['defender']) < self.capture_radius
        
        reward = jax.lax.cond(
            done_win,
            lambda: 200.0,
            lambda: jax.lax.cond(
                done_loss,
                lambda: -200.0,
                lambda: -((dist_goal - self.goal_radius) ** 2)
            )
        )

        done = jnp.logical_or(done_win, done_loss)
        return reward, done
    



    def step(self, state, action, player):
        next_state, next_nn_state= self._update_state(state, action, player)
        reward, done = self._get_reward_done(state)
        g = self._get_contraint(state)
        return next_state, next_nn_state, reward, done, g



    
    def encode_helper(self, state):
        dist_goal = jnp.linalg.norm(state['attacker'] - self.goal_position)
        dist_capture = jnp.linalg.norm(state['attacker'][:2] - state['defender'][:2])

        return jnp.array([*state['attacker'], *state['defender'], dist_goal, dist_capture])


       
    

   



    def render(self, states, dones, mode='human', close=False):


        # Initialize lists to store past positions for drawing
        attacker_positions = []
        defender_positions = []

        for i, state in enumerate(states):
            plt.figure()
            plt.clf()

            ax = plt.gca()
            xlim = max(self.size, max(abs(state['attacker'][0]), abs(state['defender'][0])))
            ylim = max(self.size, max(abs(state['attacker'][1]), abs(state['defender'][1])))

            plt.xlim([-xlim, xlim])
            plt.ylim([-ylim, ylim])

            attacker_pos = state['attacker'][:2]
            defender_pos = state['defender'][:2]

            # Append current positions for tracking
            attacker_positions.append(attacker_pos)
            defender_positions.append(defender_pos)

            # Drawing past positions as dots
            for pos in attacker_positions:
                ax.plot(*pos, 'b.', markersize=3)
            for pos in defender_positions:
                ax.plot(*pos, 'r.', markersize=3)

            # Drawing attackers and defenders
            attacker = plt.Circle(attacker_pos, 0.1, color='b', fill=True)
            defender = plt.Circle(defender_pos, self.capture_radius, color='r', fill=True)
            defender_capture_radius = plt.Circle(defender_pos, self.capture_radius + self.v_max, color='r', fill=False, linestyle='--')

            goal = plt.Circle(self.goal_position, self.goal_radius, color='g', fill=False)

            ax.add_artist(attacker)
            ax.add_artist(defender)
            ax.add_artist(defender_capture_radius)
            ax.add_artist(goal)

            # Check if done
            if dones[i]:
                break

            # Save current figure to images list
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.images.append(np.array(imageio.v2.imread(buf)))
            plt.close()



    def make_gif(self, file_name='two_player_animation_pg.gif'):

        end_state = self.images[-1]
        imageio.mimsave(file_name, self.images, fps=30)
        self.images = []
        self.positions = {'attacker': [], 'defender': []}

        return end_state
