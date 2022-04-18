import numpy as np

from mdp import EnvironmentMDP


def generate_grid_environment(shape=(8, 8), traps_rate=0.1, treasure_position=None):
    """
    Args:
        shape: grid shape (rows x columns)
        traps_rate: probability of adding a trap cell
        treasure_position: position of the treasure (default None: assigns randomly)

    Returns:

    """
    rows, cols = shape
    matrix = np.zeros(shape)
    # adding borders
    matrix[0, :] = 2
    matrix[-1, :] = 2
    matrix[:, 0] = 2
    matrix[:, -1] = 2
    # adding blocks
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if np.random.rand() < traps_rate:
                matrix[r, c] = 1
    # adding treasure
    if treasure_position is None:
        r = np.random.choice(np.arange(1, rows - 1))
        c = np.random.choice(np.arange(1, cols - 1))
    else:
        r, c = treasure_position
    matrix[r, c] = 3
    return matrix


class GridEnv(EnvironmentMDP):
    # Rewards
    NORMAL = 0
    TRAP = 1
    CLIFF = 2
    TREASURE = 3

    REWARDS = {0: -1,  # white cell (TILE)
               1: -5,  # cyan cell (TRAP)
               2: -10,  # blue cell (CLIFF)
               3: 30}  # green cell (TREASURE)

    # Actions
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    EXIT = 4

    LANDING_PROBABILITY = 0.7
    NOISE = 0.3

    def __init__(self, matrix):
        self.transition_probabilities = [self.LANDING_PROBABILITY, self.NOISE / 2, self.NOISE / 2]
        self.matrix = np.array(matrix)
        self.rows, self.columns = self.matrix.shape
        self.actions = [self.NORTH, self.EAST, self.SOUTH, self.WEST]

    def get_states(self):
        return [(i, j) for i in range(self.rows) for j in range(self.columns)]

    def get_actions(self, state):
        if self.is_terminal(state):  # this is a terminal state
            return [self.EXIT]
        return self.actions

    def get_state_reward(self, state):
        return self.REWARDS[self.matrix[state]]

    def get_s_primes(self, state, action):
        if action == self.NORTH:
            s_primes = [self.north(state), self.east(state), self.west(state)]
        elif action == self.EAST:
            s_primes = [self.east(state), self.north(state), self.east(state)]
        elif action == self.SOUTH:
            s_primes = [self.south(state), self.east(state), self.west(state)]
        elif action == self.WEST:
            s_primes = [self.west(state), self.north(state), self.east(state)]
        else:  # action == self.EXIT:
            return None
        return list(zip(s_primes, self.transition_probabilities))

    def get_initial_state(self):
        while True:
            r = np.random.choice(np.arange(1, self.rows - 1))
            c = np.random.choice(np.arange(1, self.columns - 1))
            if not self.is_terminal((r, c)):
                return r, c

    def is_terminal(self, state):
        return self.matrix[state] in {self.TREASURE, self.CLIFF}

    def north(self, state):
        row, column = state
        # If at top edge, can't move up
        if row > 0:
            row -= 1
        return row, column

    def south(self, state):
        row, column = state
        # If at bottom edge, can't move down
        if row < self.rows - 1:
            row += 1
        return row, column

    def east(self, state):
        row, column = state
        # If at right edge, can't move right
        if column < self.columns - 1:
            column += 1
        return row, column

    def west(self, state):
        row, column = state
        # If at left edge, can't move left
        if column > 0:
            column -= 1
        return row, column

    def policy_walk(self, policy, start, noise=False, max_steps=100):
        state = start
        action = policy[start]
        path = [start]
        r = 0
        while action != self.EXIT and len(path) < max_steps:
            # [(s_prime1, probability1), (s_prime2, probability2), ...]
            s_primes = self.get_s_primes(state, action)
            if noise:
                states = []
                probs = []
                for s, prob in s_primes:
                    states.append(s)
                    probs.append(prob)
                s_prime = states[np.random.choice(np.arange(len(states)), p=probs)]
            else:
                s_prime = s_primes[0][0]
            path.append(s_prime)
            r += self.get_state_reward(s_prime)
            action = policy[s_prime]
            state = s_prime
        return path, r
