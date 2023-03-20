import rule_extention
import mcts
import matplotlib.pyplot as plt
# imports from standard libs
import networkx as nx

# chrono imports
import pychrono as chrono

from control_optimisation import create_grab_criterion_fun, create_traj_fun, get_object_to_grasp
from rostok.graph_grammar.graph_grammar import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.virtual_experiment_chrono.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
from rostok.utils.pickle_save import load_saveable
from rostok.graph_generators.mcts_helper import MCTSHelper, MCTSSaveable

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
path = r"results\Reports_23y_03m_20d_02H_22M\MCTS_data.pickle"
report: MCTSSaveable = load_saveable(path)

# %% Create condig optimizing control


WEIGHT = [5, 10, 2]

cfg = ConfigRewardFunction()
cfg.bound = (2, 10)
cfg.iters = 5
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.005
cfg.time_sim = 15
cfg.flags = [FlagMaxTime(15), FlagNotContact(15), FlagSlipout(15, 15)]
"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)

best_graph, reward, best_control = report.get_best_info()
print('Best Graph is ', best_graph)
print('Best Control is ', best_control)
print('Reward is ', reward)

best_control = [float(x) for x in best_control]
func_reward = control_optimizer.create_reward_function(best_graph)
best_control= [0.1]
res = -func_reward(best_control, True)
# plot_graph(best_graph)
print(res)