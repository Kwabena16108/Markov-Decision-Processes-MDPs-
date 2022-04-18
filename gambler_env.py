import numpy as np
from mdp import EnvironmentMDP


class GamblerEnv(EnvironmentMDP):
    """
    A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips.
    If the coin comes up heads, he wins as many dollars as he has staked on that flip;
    if it is tails, he loses his stake.
    The game ends when the gambler wins by reaching his goal of $100, or loses by running out of money.
    """

    def __init__(self, goal=1000, p_h=0.5):
        """
        Args:
            goal: The goal the gambler is trying to reach
            p_h: The probability of the coin coming up heads
        """
        self.goal = goal
        self.p_h = p_h

    def get_states(self):
        return np.arange(0, self.goal + 1)

    def get_actions(self, state):
        """
        On each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars.
        Args:
            state: current capital

        Returns: amount to stake
        """
        if self.is_terminal(state):
            return [0]
        return np.arange(1, min(state, self.goal - state) + 1)

    def get_state_reward(self, state):
        """
        The reward is zero on all transitions except:
        +1 when the goal is reached
        -1 when the gambler runs out of capital to stake
        Args:
            state: current capital

        Returns:
            reward associated with that state
        """
        if state >= self.goal:
            return 1
        elif state <= 0:
            return -1
        return 0

    def get_s_primes(self, state, action):
        """
        Args:
            state:
            action:

        Returns:
            a zipped list of each possible s_prime and the corresponding landing probability
        """
        if self.is_terminal(state):
            return None
        return zip([state + action, state - action], [self.p_h, 1 - self.p_h])

    def get_initial_state(self):
        """
        Returns: initial state
        """
        return np.random.randint(0, self.goal // 2)

    def is_terminal(self, state):
        return state <= 0 or state >= self.goal
