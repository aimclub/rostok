import matplotlib.pyplot as plt
import mcts
# imports from standard libs
import networkx as nx
# chrono imports
import pychrono as chrono
import ruleset
from control_optimisation import (create_grab_criterion_fun, create_traj_fun,
                                  get_object_to_grasp)

import rostok.graph_generators.graph_environment as env
from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact,
                                                FlagSlipout)
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (
    ConfigRewardFunction, ControlOptimizer)
#from rostok.utils.result_saver import read_report
from rostok.graph_grammar.graph_utils import plot_graph, plot_graph_ids


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    #plt.savefig("./results/graph.jpg")
    plt.show()


# %% Create extension rule vocabulary

# %% Create extension rule vocabulary

rule_vocabul, node_features = ruleset.create_rules()

# %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [3, 1, 1, 2]

cfg = ConfigRewardFunction()
cfg.bound = (2, 10)
cfg.iters = 5
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.005
cfg.time_sim = 1
cfg.flags = [FlagMaxTime(1), FlagNotContact(1), FlagSlipout(0.5, 0.5)]
"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)
G = GraphGrammar()
# ,"TerminalEndLimb1","TerminalEndLimb1", "TerminalEndLimb1", "TerminalFlat1", "TerminalTransformRight1","TerminalRoundTransform","TerminalRoundTransform"
#rules = ["InitMechanism", "Add_First_Mount", "FirstLink","FirstLink", "FingerUpper"]
rules = ["InitMechanism", "Add_First_Mount", "Add_Mount", "Add_Mount", "Add_Mount", "Add_Mount"]
for rule in rules:
    G.apply_rule(rule_vocabul.get_rule(rule))
    #plot_graph(G)

plot_graph_ids(G)
print(G.get_root_based_paths())
rule_vocabul.make_graph_terminal(G)
#plot_graph(G)
result_optimizer = control_optimizer.start_optimisation(G)
print(result_optimizer[0])
cfg = ConfigRewardFunction()
cfg.bound = (2, 10)
cfg.iters = 5
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.005
cfg.time_sim = 100
cfg.flags = [FlagMaxTime(100), FlagNotContact(100), FlagSlipout(100, 100)]
"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)
func_reward = control_optimizer.create_reward_function(G)
res = - func_reward(result_optimizer[1], True)
print(res)
