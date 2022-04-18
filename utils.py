import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time


def gamma_theta_convergence(grid_mdp, mode, gammas, thetas, max_iter=1000):
    iterations_matrix = np.zeros((len(gammas), len(thetas)))
    runtimes = np.zeros_like(iterations_matrix)
    for i, gamma in enumerate(gammas):
        for j, theta in enumerate(thetas):
            s = time.time()
            if mode == "value":
                iteration, _, _, _ = grid_mdp.value_iteration(theta=theta, gamma=gamma, max_iter=max_iter)
            elif mode == "policy":
                iteration, _, _, _ = grid_mdp.policy_iteration(theta=theta, gamma=gamma, max_iter=max_iter)
            else:
                raise Exception("Mode not recognized. Expects 'value' or 'policy'.")
            runtime = time.time() - s
            iterations_matrix[i, j] = iteration
            runtimes[i, j] = runtime
    return iterations_matrix, runtimes

def gamma_alpha_convergence(grid_mdp, gammas, alphas, max_episode=10000):
    runs_matrix = np.zeros((len(gammas), len(alphas)))
    runtimes = np.zeros_like(runs_matrix)
    for i, gamma in enumerate(gammas):
        for j, alpha in enumerate(alphas):
            s = time.time()
            _, scores, _ = grid_mdp.q_learning(gamma=gamma, learning_rate=alpha)
            runtime = time.time() - s
            runs_matrix[i, j] = np.mean(scores)
            runtimes[i, j] = runtime
    return runs_matrix, runtimes


def plot_heatmap(matrix, title, xlabel="θ", ylabel="γ", reverse_cmap=False, annot=True,
                 *args, **kwargs):
    """
    Args:
        matrix: 2-D numerical data
        title: Title of the plot
        xlabel: label on the x-axis
        ylabel: label on the y-axis
        reverse_cmap: bool, default False
        annot: show values in the heatmap grid
        *args: other arguments to pass to `seaborn.heatmap`
        **kwargs: other keyword arguments to pass to `seaborn.heatmap`

    Returns:

    """
    plt.figure(figsize=(12, 12))
    color_map = plt.cm.get_cmap("RdYlGn")
    if reverse_cmap:
        color_map = color_map.reversed()
    sns.heatmap(matrix, annot=annot, cmap=color_map, annot_kws={"fontsize": 12},
                fmt='.2g' if isinstance(annot, bool) else '', *args, **kwargs)
    plt.xlabel(xlabel, fontsize=22)
    plt.xticks(fontsize=12)
    plt.ylabel(ylabel, fontsize=22)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=24)
