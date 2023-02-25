from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import mcts
# imports from standard libs
import networkx as nx
import optmizers_config
# chrono imports
import pychrono as chrono
from obj_grasp.objects import (get_obj_easy_box, get_obj_hard_ellipsoid, get_object_to_grasp_sphere,
    get_obj_hard_large_ellipsoid, get_obj_easy_large_box, get_obj_hard_long_ellipsoid,get_obj_hard_long_tilt_ellipsoid, get_obj_easy_long_tilt_box)
from rule_sets import rule_extention_graph
from rule_sets.ruleset_old_style_graph import create_rules

from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact,
                                                FlagSlipout)
from rostok.graph_generators.mcts_helper import (make_mcts_step,
                                                 prepare_mcts_state_and_helper)
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from rostok.utils.pickle_save import load_saveable
from rostok.graph_generators.mcts_helper import OptimizedGraphReport


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()


graph_report: OptimizedGraphReport = load_saveable(Path(r"results\Reports_23y_02m_26d_00H_53M\optimized_graph_report.pickle"))
# %% Create extension rule vocabulary
rule_vocabul, torque_dict = create_rules()
#rule_vocabul = deepcopy(rule_extention_graph.rule_vocab)
#torque_dict = rule_extention_graph.torque_dict
cfg = optmizers_config.get_cfg_graph(torque_dict)
#cfg.get_rgab_object_callback = get_obj_hard_large_ellipsoid
#cfg.get_rgab_object_callback = get_obj_easy_box
#cfg.get_rgab_object_callback =get_object_to_grasp_sphere
cfg.get_rgab_object_callback = get_obj_easy_long_tilt_box
control_optimizer = ControlOptimizer(cfg)

graph_list = graph_report.graph_list
reward_list = []
i_list = set()

# for graph in graph_list:
#     reward_list.append(graph.reward)
#     i_list.add(int(graph.reward))

# print(sorted(i_list))
# plt.hist(reward_list)
# plt.show()
# for graph in graph_list:
#     if graph.reward >5.6 and graph.reward < 5.7:
#         G = graph.graph
#         reward = graph.reward
#         control = graph.control
#         break
#     #print(graph.reward)


# func_reward = control_optimizer.create_reward_function(G)
# res = -func_reward(control, True)
# print(res)
# print(reward)

#best_graph, reward, best_control = report.get_best_info()

top_list =[]
sorted_graph_list = sorted(graph_list, key = lambda x: x.reward)
ten_top = sorted_graph_list[-1:-11:-1]
for graph in ten_top:
    G = graph.graph
    reward = graph.reward
    control = graph.control
    func_reward = control_optimizer.create_reward_function(G)
    res = -func_reward(control, True)
    print(reward)
    print(res)
    print()


