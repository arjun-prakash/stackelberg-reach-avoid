import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib
import io
import imageio
matplotlib.use('Agg')

import jax.numpy as jnp

class TwoPlayerGridWorldEnv:
    def __init__(self, game_type, grid_size, init_attacker_position, init_defender_position, goal_position, capture_radius):
        self.players = ['attacker', 'defender']

        self.game_type = game_type
        self.grid_size = grid_size
        self.init_attacker_position = init_attacker_position
        self.init_defender_position = init_defender_position
        self.goal_position = goal_position
        self.goal_radius = 1
        self.capture_radius = capture_radius
        self.num_actions = 4  # Up, Down, Left, Right
        self.max_steps=50

        self.state = {
            'attacker': np.array(self.init_attacker_position),
            'defender': np.array(self.init_defender_position)
        }

        self.images = []


    def step(self, state, action, player, update_env=False):
        next_state = state.copy()
        if action == 0:  # Up
            next_state[player][1] -= 1
        elif action == 1:  # Down
            next_state[player][1] += 1
        elif action == 2:  # Left
            next_state[player][0] -= 1
        elif action == 3:  # Right
            next_state[player][0] += 1

        # Wrap around logic
        next_state[player] = next_state[player] % self.grid_size
        distance_to_goal = np.linalg.norm(next_state['attacker'] - self.goal_position)

        reward = -(distance_to_goal**2)
        done = False
        info = {'player': player, 'is_legal':True, 'status':'running'}


        #if attacker reaches goal
        if distance_to_goal < self.goal_radius:
            if player == 'attacker':
                info = {'player': player, 'is_legal':True, 'status':'attacker reached goal'}
            reward = 100  # Attacker wins
            done = True

        elif np.linalg.norm(next_state['attacker'] - next_state['defender']) <= self.capture_radius:
            if player == 'attacker':
                info = {'player': player, 'is_legal':False, 'status':'attacker collided with defender'}
            #reward = -1  # Defender wins
            done = True

        if update_env:
            self.state = next_state

        return next_state, reward, done, info

    def reset(self, key=None):


        self.state = {
            'attacker': np.array(self.init_attacker_position),
            'defender': np.array(self.init_defender_position)
        }

        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 1e6))

        # Split the key for attacker and defender
        key, subkey1, subkey2 = jax.random.split(key, 3)

        attacker_pos = jax.random.randint(subkey1, (2,), 0, self.grid_size)
        defender_pos = jax.random.randint(subkey2, (2,), 0, self.grid_size)

        # Convert JAX arrays to NumPy arrays
        self.state['attacker'] = np.array(attacker_pos)
        self.state['defender'] = np.array(defender_pos)

        # Ensure that the defender and attacker are at least 2 steps away from each other
        while np.linalg.norm(self.state['attacker'] - self.state['defender']) <= self.capture_radius:
            # generate new subkeys for attacker and defender
            key, subkey1, subkey2 = jax.random.split(subkey1, 3)

            attacker_pos = jax.random.randint(subkey1, (2,), 0, self.grid_size)
            defender_pos = jax.random.randint(subkey2, (2,), 0, self.grid_size)

            self.state['attacker'] = np.array(attacker_pos)
            self.state['defender'] = np.array(defender_pos)

            # ensure that the attacker is at least 2 steps away from the goal
            while np.linalg.norm(self.state['attacker'] - self.goal_position) <= self.goal_radius + 3:
                key, subkey1, subkey2 = jax.random.split(subkey1, 3)

                attacker_pos = jax.random.randint(subkey1, (2,), 0, self.grid_size)
                self.state['attacker'] = np.array(attacker_pos)

        return self.state

    

    def set(self, ax, ay, dx, dy):
        """
        Manually set the environment to a specific state.
        ax, ay: Attacker's x and y positions
        dx, dy: Defender's x and y positions
        """
        self.state['attacker'] = np.array([ax, ay])
        self.state['defender'] = np.array([dx, dy])

        # Optionally, you can also add logic to set other state variables if needed
        # For example, if you have a time step or other parameters in the state, set them here

        return self.state


    def set_to(self, state):
            """
            Manually set the environment to a specific state.
            ax, ay: Attacker's x and y positions
            dx, dy: Defender's x and y positions
            """
            self.state = state

            # Optionally, you can also add logic to set other state variables if needed
            # For example, if you have a time step or other parameters in the state, set them here

            return self.state

    def render(self, mode='human', close=False):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)

        # Draw grid lines
        ax.set_xticks(np.arange(0, self.grid_size, 1))
        ax.set_yticks(np.arange(0, self.grid_size, 1))
        ax.grid(which='both')

        # Draw players and goal
        ax.plot(self.goal_position[0], self.goal_position[1], 'gs', markersize=15)  # Goal
        ax.plot(self.state['attacker'][0], self.state['attacker'][1], 'bs', markersize=10)  # Attacker
        ax.plot(self.state['defender'][0], self.state['defender'][1], 'rs', markersize=10)  # Defender

        # Capture the plot as an image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.images.append(image)

        plt.close(fig)

    def make_gif(self, file_name='gifs\debug\two_player_gridworld.gif'):
        imageio.mimsave(file_name, self.images, fps=10)
        self.images = []  # Clear the image list for the next episode


    def unconstrained_select_action(self, nn_state, params, policy_net, key, epsilon):
        if jax.random.uniform(key) < epsilon:
            return jax.random.choice(key, self.num_actions)
        else:
            probs = policy_net.apply(params, nn_state)
            return jax.random.choice(key, a=self.num_actions, p=probs)
        
    def get_legal_actions_mask(self, state, player):
        legal_actions_mask = []
        for action in range(self.num_actions):
            _, _, _, info = self.step(state, action,player, update_env=False)
            legal_actions_mask.append(int(info['is_legal']))
        return jnp.array(legal_actions_mask)
    
    def constrained_select_action(self, nn_state, policy_net, params, legal_actions_mask, key, epsilon):
        if jax.random.uniform(key) < epsilon:
            legal_actions_indices = jnp.arange(len(legal_actions_mask))[legal_actions_mask.astype(bool)]
            return jax.random.choice(key, legal_actions_indices)
        else:
            probs = policy_net.apply(params, nn_state, legal_actions_mask)
            #action = jax.random.categorical(key, probs)
            action = jax.random.choice(key, a=self.num_actions, p=probs)
            return action
    


    def single_rollout(self,args):
        #print("env state" , self.state)
        game_type, params, policy_net, key, epsilon, gamma, render, for_q_value, one_step_reward, state = args
        if game_type != self.game_type:
            raise ValueError(f"game_type {game_type} does not match self.game_type {self.game_type}")


        states = {player: [] for player in self.players}
        actions = {player: [] for player in self.players}
        action_masks = {player: [] for player in self.players}
        rewards = {player: [] for player in self.players}
        padding_mask = {player: [] for player in self.players}

        wins = {'attacker': 0, 'defender': 0, 'draw': 0}
        done = False
        step = 0
        defender_wins = False
        attacker_wins = False
        defender_oob = False

        defender_no_legal_moves = False
        attacker_no_legal_moves = False

        if state == None:
            state = self.state
        #state = self.reset()
        else:
            self.state = state

        if for_q_value:
            #append 0 to rewards
            rewards['attacker'].append(one_step_reward)
            rewards['defender'].append(0)
            step = 1
        # else: 
        #     state = self.reset()
            
        nn_state = self.encode_helper(state)


        while not done and not defender_wins and not attacker_wins and step < self.max_steps:
            for player in self.players:
                states[player].append(nn_state)

                key, subkey = jax.random.split(key)


                if game_type == 'nash':
                    action = self.unconstrained_select_action(nn_state, params[player], policy_net,  subkey, epsilon)
                    state, reward, done, info = self.step(state=state, action=action, player=player, update_env=True)
                    nn_state = self.encode_helper(state)
                    actions[player].append(action)
                    rewards[player].append(reward)
                    action_masks[player].append([1]*self.num_actions)


                elif game_type == 'stackelberg':
                    legal_actions_mask = self.get_legal_actions_mask(state, player)
                    if sum(legal_actions_mask) != 0:
                        if player == 'defender':
                            action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)
                        elif player == 'attacker':
                            action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)

                            #action = self.constrained_deterministic_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)
                        #action = self.constrained_select_action(nn_state, policy_net, params[player], legal_actions_mask, subkey, epsilon)

                        action_masks[player].append(legal_actions_mask)
                        state, reward, done, info = self.step(state=state, action=action, player=player, update_env=True)
                        nn_state = self.encode_helper(state)
                        actions[player].append(action)
                        rewards[player].append(reward)
                    else: #case where a player has no legal moves
                        done = True
                        if player == 'defender': 
                            defender_no_legal_moves = True
                        elif player == 'attacker': 
                            attacker_no_legal_moves = True

                            
                
                
                if render:
                    self.render()



            step += 1
            #print(step)

        if not defender_wins and not attacker_wins:
            wins['draw'] = 1
            #rewards['attacker'][-1] = -1

        returns = {player: [] for player in self.players}

        for player in self.players:
            G = 0
            for r in reversed(rewards['attacker']):
                G = r + gamma * G
                returns[player].append(G)
            
            returns[player] = list(reversed(returns[player]))

        if not for_q_value:
            for player in self.players:
                states[player], actions[player], action_masks[player], returns[player], padding_mask[player] = self.pad_and_mask(states[player], actions[player], action_masks[player], returns[player])
    
        

        return states, actions, action_masks, returns, padding_mask, wins


# # Example usage
# env = TwoPlayerGridWorldEnv(grid_size=5, init_attacker_position=[0, 0], init_defender_position=[4, 4], goal_position=[2, 2], capture_radius=1)
# state = env.reset()
# env.render()
# next_state, reward, done, info = env.step(state, 1, 'attacker', update_env=True)
# env.render()

    def encode_helper(self, env_state):
        # Normalize the coordinates to be between 0 and 1
        max_value = self.grid_size - 1
        attacker_x, attacker_y = env_state['attacker'] / max_value
        defender_x, defender_y = env_state['defender'] / max_value

        #distance to goal
        distance_to_goal = np.linalg.norm(env_state['attacker'] - self.goal_position) / max_value

        #distance between attacker and defender
        distance_between_players = np.linalg.norm(env_state['attacker'] - env_state['defender']) / max_value



        # Flatten and combine the state into a single array
        encoded_state = np.array([attacker_x, attacker_y, defender_x, defender_y, distance_to_goal, distance_between_players])

        return encoded_state

    def pad_and_mask(self, states, actions, action_masks, returns):
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        action_masks = np.array(action_masks)
        returns = np.array(returns)

        # Pad the states, actions, and returns
        padded_states = np.pad(states, ((0, self.max_steps - len(states)), (0, 0)), 'constant', constant_values=0)
        padded_actions = np.pad(actions, (0, self.max_steps  - len(actions)), 'constant', constant_values=0)
        padded_action_masks = np.pad(action_masks, ((0, self.max_steps  - len(action_masks)), (0,0)), 'constant', constant_values=0)
        padded_returns = np.pad(returns, (0, self.max_steps  - len(returns)), 'constant', constant_values=0)

        # Create a mask where the value is 1 for actual values and 0 for padding
        padding_mask = np.concatenate([np.ones(len(states)), np.zeros(self.max_steps  - len(states))])

        return padded_states, padded_actions, padded_action_masks, padded_returns, padding_mask



