import rule_extention
import mcts
import matplotlib.pyplot as plt
# imports from standard libs
import networkx as nx

# chrono imports
import pychrono as chrono

from control_optimisation import create_grab_criterion_fun, create_traj_fun, get_pipes
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
from rostok.utils.result_saver  import read_report

import rostok.graph_generators.graph_environment as env


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()

# %% Create extension rule vocabulary

rule_vocabul, node_features = rule_extention.init_extension_rules()

# %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [3, 1, 1, 2]

cfg = ConfigRewardFunction()
cfg.bound = (2, 10)
cfg.iters = 5
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.005
cfg.time_sim = 2
cfg.flags = [FlagMaxTime(2), FlagNotContact(1), FlagSlipout(0.5, 0.5)]

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_pipes
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)