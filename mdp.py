from abc import *
import numpy as np


class EnvironmentMDP:
    @abstractmethod
    def get_states(self):
        """
        Returns: an iterable of all possible states

        """
        pass

    @abstractmethod
    def get_actions(self, state):
        """
        Args:
            state:

        Returns:
            possible actions from that state
        """
        pass

    @abstractmethod
    def get_state_reward(self, state):
        """
        Args:
            state:

        Returns:
            reward associated with that state
        """
        pass

    @abstractmethod
    def get_s_primes(self, state, action):
        """
        Args:
            state:
            action:

        Returns:
            a zipped list of each possible s_prime and the corresponding landing probability
        """
        pass

    @abstractmethod
    def get_initial_state(self):
        """
        Returns: initial state
        """
        pass

    @abstractmethod
    def is_terminal(self, state):
        """
        Args:
            state:

        Returns:
            bool: if the state is terminal state
        """
        pass

    def take_step(self, state, action):
        s_primes = self.get_s_primes(state, action)
        states = []
        probs = []
        for s, prob in s_primes:
            states.append(s)
            probs.append(prob)
        return states[np.random.choice(np.arange(len(states)), p=probs)]


class MDP:
    def __init__(self, environment: EnvironmentMDP):
        self.environment = environment

    def calculate_v_value(self, state, action, v_values, gamma):
        s_primes = self.environment.get_s_primes(state, action)
        if s_primes is None:  # state is terminal
            return self.environment.get_state_reward(state)
        v_value = 0
        for s_prime, probability in s_primes:
            r = self.environment.get_state_reward(s_prime)
            discounted_v = gamma * v_values[s_prime]
            v_value += probability * (r + discounted_v)
        return v_value

    def get_optimal_action(self, state, v_values, gamma):
        best_v_sa = float("-inf")
        best_action = None
        for action in self.environment.get_actions(state):
            # V(s, a) for each action a on the current state s
            v_sa = self.calculate_v_value(state, action, v_values, gamma)
            # get the action a with the best V(s, a)
            if v_sa > best_v_sa:
                best_v_sa = v_sa
                best_action = action
        return best_action, best_v_sa

    def policy_evaluation(self, pi, v_values, theta, gamma, max_iter=1000):
        iteration = 0
        deltas = []
        policy = {} if pi is None else pi
        while iteration < max_iter:
            delta = 0
            for s in v_values.keys():
                v = v_values[s]
                if pi is None:  # find best action in value iteration
                    a, v_sa = self.get_optimal_action(s, v_values, gamma)
                    v_values[s] = v_sa
                    policy[s] = a
                else:  # evaluate the given policy
                    a = pi[s]
                    v_values[s] = self.calculate_v_value(s, a, v_values, gamma)
                delta = max(delta, abs(v_values[s] - v))
            iteration += 1
            deltas.append(delta)
            if delta < theta:
                break

        return iteration, policy, deltas

    def value_iteration(self, theta, gamma, max_iter=1000):
        """
        Args:
            theta: convergence threshold
            gamma: discount rate
            max_iter: maximum number of iterations

        Returns:
            iteration: number of iterations until convergence
            pi: optimal policy
            v_values: dictionary state->v_value
        """
        v_values = {}
        for state in self.environment.get_states():
            # initialize V: for all states s, V(s) = 0
            v_values[state] = 0

        iteration, pi, deltas = self.policy_evaluation(pi=None,
                                                       v_values=v_values,
                                                       theta=theta,
                                                       gamma=gamma,
                                                       max_iter=max_iter)
        return iteration, pi, deltas, v_values

    def policy_improvement(self, v_values, pi, gamma):
        policy_stable = True
        for s in v_values.keys():
            old_action = pi[s]
            best_a, v_sa = self.get_optimal_action(s, v_values, gamma)
            pi[s] = best_a
            # is the policy for all previously visited states stable AND policy of the current state is also stable
            policy_stable = policy_stable and best_a == old_action
        return policy_stable

    def policy_iteration(self, theta, gamma, max_iter=1000):
        """
        Args:
            theta: a small positive number determining the accuracy of estimation
            gamma: discount rate
            max_iter: maximum number of iterations

        Returns:
            iteration: number of iterations until convergence
            pi: optimal policy
            v_values: dictionary state->v_value
        """
        assert theta > 0
        v_values = {}
        pi = {}

        # Initialization
        for state in self.environment.get_states():
            # initialize V: for all states s, V(s) = 0
            v_values[state] = 0
            pi[state] = np.random.choice(self.environment.get_actions(state))

        deltas = []
        iteration = 0
        has_converged = False
        while iteration < max_iter and not has_converged:
            _, _, d = self.policy_evaluation(pi, v_values, theta, gamma, max_iter // 10)
            deltas.append(d[-1])
            has_converged = self.policy_improvement(v_values, pi, gamma)
            iteration += 1

        return iteration, pi, deltas, v_values

    def epsilon_greedy_policy(self, state, q_state, epsilon):
        if np.random.random() < epsilon:  # EXPLORE: pick random action
            return np.random.choice(self.environment.get_actions(state))
        best_a, q_s_a = max(q_state.items(), key=lambda x: x[1])
        return best_a

    def q_learning(self, gamma=1,
                   num_episodes=1000,
                   learning_rate=1., learning_rate_decay=0.99,
                   epsilon=1., epsilon_decay=0.99):

        q = {}
        v_values = {}
        for state in self.environment.get_states():
            # initialize Q: for all state-action pairs
            q.setdefault(state, {})
            for action in self.environment.get_actions(state):
                q[state][action] = 0
            v_values[state] = 0

        scores = []
        for i in range(num_episodes):
            score = 0
            # play an episode
            state = self.environment.get_initial_state()
            while not self.environment.is_terminal(state):
                action = self.epsilon_greedy_policy(state, q[state], epsilon)
                new_state = self.environment.take_step(state, action)
                reward = self.environment.get_state_reward(new_state)
                score += reward
                # updating Q table
                if self.environment.is_terminal(new_state):
                    q[state][action] += learning_rate * (reward - q[state][action])
                else:
                    q[state][action] += learning_rate * (
                            reward + gamma * max(q[new_state].values()) - q[state][action]
                    )
                state = new_state
            scores.append(score)
            epsilon = max(0.01, epsilon * epsilon_decay)
            learning_rate = max(0.01, learning_rate * learning_rate_decay)

        # initialize state value function and optimal policy
        v_values = {}
        pi = {}
        for state in q:
            best_a, q_s_a = max(q[state].items(),
                                key=lambda x: x[1])  # {action -> value of that action from that state}
            v_values[state] = self.environment.get_state_reward(state) \
                if self.environment.is_terminal(state) else q_s_a
            pi[state] = best_a
        return pi, scores, v_values
