# imports from our code
from control_optimisation import create_grab_criterion_fun, create_traj_fun, get_object_to_grasp
import context
import matplotlib.pyplot as plt
# imports from standard libs
import networkx as nx
import numpy as np
# chrono imports
import pychrono as chrono
from pychrono import (Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z, Q_ROTATE_Y_TO_X,
                      Q_ROTATE_Y_TO_Z, Q_ROTATE_Z_TO_X, Q_ROTATE_Z_TO_Y,
                      ChCoordsysD, ChQuaternionD, ChVectorD)

from engine import node_vocabulary, rule_vocabulary
from engine.node import ROOT, BlockWrapper, GraphGrammar
from engine.node_render import ChronoBody, ChronoRevolveJoint, ChronoTransform
from utils.control_optimizer import ConfigRewardFunction, ControlOptimizer
from utils.flags_simualtions import FlagMaxTime
from utils.transform_srtucture import FrameTransform
import rule_extention
import app_vocabulary
import mcts
import stubs.graph_environment as env


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(graph, dim=2), node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()

# %% Create extension rule vocabulary

rule_vocabul = rule_extention.init_extension_rules()

# %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [1, 1, 1, 1]

cfg = ConfigRewardFunction()
cfg.bound = (-5, 5)
cfg.iters = 2
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.001
cfg.time_sim = 2
cfg.flags = [FlagMaxTime(2)]

"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(app_vocabulary.node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)

# %% Init mcts parameters

G = GraphGrammar()
max_numbers_rules = 5

# Hyperparameters mcts
iteration_limit = 10

# Initilize MCTS
searcher = mcts.mcts(iterationLimit=iteration_limit)
finish = False

G = GraphGrammar()
max_numbers_rules = 5
# Create graph envirenments for algorithm (not gym)
graph_env = env.GraphVocabularyEnvironment(G, rule_vocabul, max_numbers_rules)

graph_env.set_control_optimizer(control_optimizer)

# %% Run first algorithm
iter = 0
while not finish:
    action = searcher.search(initialState=graph_env)
    finish, final_graph, opt_trajectory = graph_env.step(action,False)
    iter +=1
    print(f"number iteration: {iter}, counter actions: {graph_env.counter_action}, reward: {graph_env.reward}")
    
# %%

plt.figure()
nx.draw_networkx(final_graph, pos=nx.kamada_kawai_layout(final_graph, dim=2), node_size=800,
                 labels={n: final_graph.nodes[n]["Node"].label for n in final_graph})

plt.show()

func_reward = control_optimizer.create_reward_function(final_graph)
func_reward(opt_trajectory, True)