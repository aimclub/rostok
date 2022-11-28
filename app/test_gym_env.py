from control_optimisation import create_grab_criterion_fun, create_traj_fun, get_object_to_grasp
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
import rule_extention
import mcts
import rostok.envs.gym_graph_enviroment as env
import matplotlib.pyplot as plt

# imports from standard libs
import networkx as nx

# chrono imports
import pychrono as chrono

rule_vocabul, node_features = rule_extention.init_extension_rules()

# %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [1, 1, 1, 1]

cfg = ConfigRewardFunction()
cfg.bound = (0, 10)
cfg.iters = 10
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.0015
cfg.time_sim = 3
cfg.flags = [FlagMaxTime(3), FlagNotContact(1), FlagSlipout(1, 0.5)]
"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)

# Create graph envirenments for algorithm (not gym)
graph_env = env.GraphGrammarEnv(rule_vocabul, control_optimizer)

# %% Run first algorithm
action = [0, 4, 4, 5, 20, 11, 29, 29, 30, 30, 8, 9]
for i in range(len(action)):
    #graph_env.action_space.sample()
    graph_env.step(action[i])

graph_env.reset()