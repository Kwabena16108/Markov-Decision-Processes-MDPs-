from grid_visualizer import visualize
from grid_env import GridEnv
from mdp import MDP

g = [[2, 2, 2, 2, 2, 2, 2, 2],
     [2, 0, 0, 0, 1, 0, 0, 2],
     [2, 0, 0, 1, 0, 0, 0, 2],
     [2, 0, 0, 0, 1, 0, 0, 2],
     [2, 0, 0, 0, 0, 0, 0, 2],
     [2, 0, 0, 0, 0, 0, 0, 2],
     [2, 0, 0, 0, 0, 0, 3, 2],
     [2, 2, 2, 2, 2, 2, 2, 2]]
grid = GridEnv(matrix=g)
mdp_obj = MDP(grid)
# iteration, pi, deltas, v_values = mdp_obj.value_iteration(theta=1e-5, gamma=1, max_iter=1000)
# iteration, pi, deltas, v_values = mdp_obj.policy_iteration(theta=1e-5, gamma=1, max_iter=1000)
pi, scores, v_values = mdp_obj.q_learning(gamma=1, num_episodes=10000)
# print(f"Number of iterations until convergence: {iteration}")
path, reward = grid.policy_walk(policy=pi, start=(1, 1), noise=False, max_steps=100)
print(f"Accumulated Reward {reward}")
visualize(grid.matrix, soln=path)
