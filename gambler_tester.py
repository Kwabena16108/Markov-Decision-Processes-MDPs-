from gambler_env import GamblerEnv
from mdp import MDP

gambler_obj = GamblerEnv(goal=10, p_h=0.5)
gambler_mdp = MDP(gambler_obj)
iteration, pi, deltas, v_values = gambler_mdp.value_iteration(theta=1e-5, gamma=1)
