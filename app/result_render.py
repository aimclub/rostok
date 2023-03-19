import rule_extention
import mcts
import matplotlib.pyplot as plt
# imports from standard libs
import networkx as nx

# chrono imports
import pychrono as chrono

from control_optimisation import create_grab_criterion_fun, create_traj_fun, get_object_to_grasp
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.virtual_experiment.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
from rostok.utils.result_saver  import read_report

import rostok.graph_generators.graph_environment as env

def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()

rule_vocabul, node_features = rule_extention.init_extension_rules()


# !!!! WRITE HERE THE PATH TO THE FILE WITH RESULTS !!!! #
path = "./results/MCTS_report_22y_12m_12d_22H_27M/mcts_log_.txt"
best_graph, best_control, reward = read_report(path, rule_vocabul)

# %% Create condig optimizing control


WEIGHT = [5, 10, 2]

cfg = ConfigRewardFunction()
cfg.bound = (2, 10)
cfg.iters = 5
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.005
cfg.time_sim = 2
cfg.flags = [FlagMaxTime(2), FlagNotContact(1), FlagSlipout(0.5, 0.5)]
"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)

print('Best Graph is ', best_graph)
print('Best Control is ', best_control)
print('Reward is ', reward)

best_control = [float(x) for x in best_control]
func_reward = control_optimizer.create_reward_function(best_graph)
res = -func_reward(best_control, True)
# plot_graph(best_graph)
print(res)