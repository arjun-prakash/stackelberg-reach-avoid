import numpy as np

class TileCoder:
    def __init__(self, num_tiles, num_tilings, state_bounds, action_bounds):
        """
        Initialize the tile coder.
        
        Parameters:
        - num_tiles: an array with the number of tiles for each dimension of the state space
        - num_tilings: the number of tilings to use
        - state_bounds: an array with the lower and upper bounds for each dimension of the state space
        - action_bounds: an array with the lower and upper bounds for each dimension of the action space
        """
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        
        # Calculate the size of the state and action spaces
        self.state_size = np.prod(self.num_tiles)
        self.action_size = np.prod(self.num_tiles[-1:])
        
        # Calculate the scaling factor for each dimension of the state and action spaces
        self.state_scaling = (self.num_tiles - 1) / (self.state_bounds[:, 1] - self.state_bounds[:, 0])
        self.action_scaling = (self.num_tiles[-1:] - 1) / (self.action_bounds[:, 1] - self.action_bounds[:, 0])
        
        # Initialize the transition matrix
        self.P = np.zeros((self.state_size * self.action_size, self.state_size))
        
    def get_tile(self, state, action):
        """
        Get the tile indices for a given state and action.
        
        Parameters:
        - state: the continuous state
        - action: the continuous action
        
        Returns:
        - tile_indices: an array with the tile indices for the given state and action
        """
        # Normalize the state and action
        state_normalized = (state - self.state_bounds[:, 0]) * self.state_scaling
        action_normalized = (action - self.action_bounds[:, 0]) * self.action_scaling
        
        # Calculate the tile indices
        tile_indices = np.floor(state_normalized) + np.floor(action_normalized) * self.num_tiles[:-1]
        
        # Calculate the offset for each tiling
        offset = np.arange(self.num_tilings) * self.state_size
        
        # Add the offset to the tile indices to get the final indices
        tile_indices += offset
        
        return tile_indices
    
    def update_transition_matrix(self, state, action, next_state, probability):
        """
        Update the transition matrix with the given transition information.
        
        Parameters:
        - state: the current continuous state
        - action: the current continuous action
        - next_state: the next
        state: the next continuous state
        - probability: the probability of transitioning from the current state and action to the next state
        
        """
        # Get the tile indices for the current state and action
        current_indices = self.get_tile(state, action)
        
        # Get the tile indices for the next state
        next_indices = self.get_tile(next_state, action)
        
        # Update the transition matrix
        for current_index in current_indices:
            for next_index in next_indices:
                self.P[current_index, next_index] += probability