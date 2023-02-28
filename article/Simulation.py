from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import mcts
# imports from standard libs
import networkx as nx
import optmizers_config
# chrono imports
import pychrono as chrono
from obj_grasp.objects import get_obj_easy_box, get_obj_hard_ellipsoid, get_object_to_grasp_sphere, get_obj_hard_large_ellipsoid, get_obj_easy_large_box, get_obj_hard_long_ellipsoid, get_obj_easy_long_box, get_obj_hard_large_ellipsoid
from rule_sets import rule_extention_graph
from rule_sets.ruleset_old_style_graph_nonails import create_rules

from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact,
                                                FlagSlipout)
from rostok.graph_generators.mcts_helper import (make_mcts_step,
                                                 prepare_mcts_state_and_helper)
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from rostok.utils.pickle_save import load_saveable


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()


# %% Create extension rule vocabulary
rule_vocabul, torque_dict = create_rules()
#rule_vocabul = deepcopy(rule_extention_graph.rule_vocab)
#torque_dict = rule_extention_graph.torque_dict
cfg = optmizers_config.get_cfg_graph(torque_dict)
cfg.get_rgab_object_callback = get_obj_hard_large_ellipsoid
control_optimizer = ControlOptimizer(cfg)


G = GraphGrammar()
# rules = ["Init", "AddFinger_N", "AddFinger",  "Terminal_Negative_Translate2",  "RemoveFinger_P",  "Terminal_Radial_Translate1", "RemoveFinger_RN", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Radial_Translate1", "Remove_FG", "Terminal_Link3", "AddFinger_R", "RemoveFinger_RP", "Phalanx", "Terminal_Joint5", "Terminal_Link3", "Remove_FG", "Terminal_Radial_Translate1"]
# rules = ["Init", 
#          "RemoveFinger",  
#          "RemoveFinger_N", 
#          "RemoveFinger_R", 
#          "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
#          "RemoveFinger_P",
#          "RemoveFinger_RP"
#          ]
# for rule in rules:
#     G.apply_rule(rule_vocabul.get_rule(rule))

# func_reward = control_optimizer.create_reward_function(G)
# #plot_graph(G)
# best_control = []
# res = -func_reward(best_control, True)
# print()
# print(res)
# G = GraphGrammar()
# rules = ["Init", 
#          "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link2", 
#          "RemoveFinger_N", 
#          "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link2",
#          "RemoveFinger_RN", 
#          "RemoveFinger_P",
#          "RemoveFinger_RP"
#          ]
# for rule in rules:
#     G.apply_rule(rule_vocabul.get_rule(rule))

# func_reward = control_optimizer.create_reward_function(G)
# #plot_graph(G)
# best_control = []
# res = -func_reward(best_control, True)
# print()
# print(res)

G = GraphGrammar()
rules = ["Init", 
         "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Remove_FG", "Terminal_Link3", "Terminal_Joint5", "Terminal_Link2", "Terminal_Joint2",
         "RemoveFinger_N", 
         "RemoveFinger_R", 
         "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5", "Terminal_Link3", "Terminal_Joint2", "Terminal_Link2",
         "RemoveFinger_P",
         "AddFinger_RPT","Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5",  "Terminal_Link3", "Terminal_Joint2",  "Terminal_Link2"
         ]

for rule in rules:
    G.apply_rule(rule_vocabul.get_rule(rule))

func_reward = control_optimizer.create_reward_function(G)
# plot_graph(G)
best_control = []
res = -func_reward(best_control, True)
print()
print(res)

G = GraphGrammar()
# rules = ["Init", 
#          "RemoveFinger",  
#          "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
#          "RemoveFinger_R", 
#          "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
#          "RemoveFinger_P",
#          "RemoveFinger_RP"
#          ]

# rules = ["Init", 
#          "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3", 
#          "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
#          "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
#          "RemoveFinger_RN", 
#          "RemoveFinger_P",
#          "AddFinger_RP","Terminal_Radial_Translate1", "Terminal_Positive_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3"
#          ]

# rules = ["Init", 
#          "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3", 
#          "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
#          "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
#          "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
#          "AddFinger_P","Terminal_Radial_Translate1", "Terminal_Positive_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
#          "AddFinger_RP","Terminal_Radial_Translate1", "Terminal_Positive_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3"
#          ]

rules = ["Init", 
         "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Remove_FG", "Terminal_Link3", "Terminal_Joint5", "Terminal_Link2", "Terminal_Joint2",
         "RemoveFinger_N", 
         "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint3", "Remove_FG", "Terminal_Link3",
         "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5", "Terminal_Link3", "Terminal_Joint2", "Terminal_Link2",
         "RemoveFinger_P",
         "AddFinger_RPT","Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5",  "Terminal_Link3", "Terminal_Joint2",  "Terminal_Link2"
         ]

rules = ["Init", 
         "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Remove_FG", "Terminal_Link3", "Terminal_Joint5", "Terminal_Link2", "Terminal_Joint2",
         "RemoveFinger_N", 
         "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint3", "Remove_FG", "Terminal_Link3",
         "RemoveFinger_RN", 
         "RemoveFinger_P",
         "RemoveFinger_RP"
         ]

# rules = ["Init", 
#          "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Remove_FG", "Terminal_Link3", "Terminal_Joint5", "Terminal_Link2", "Terminal_Joint2",
#          "RemoveFinger_N", 
#          "RemoveFinger_R", 
#          "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5", "Terminal_Link3", "Terminal_Joint2", "Terminal_Link2",
#          "RemoveFinger_P",
#          "AddFinger_RPT","Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5",  "Terminal_Link3", "Terminal_Joint2",  "Terminal_Link2"
#          ]

# rules = ["Init", 
#          "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Remove_FG", "Terminal_Link3", "Terminal_Joint5", 
#          "RemoveFinger_N", 
#          "RemoveFinger_R", 
#          "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx",  "Remove_FG", "Terminal_Joint5", "Terminal_Link3", 
#          "RemoveFinger_P",
#          "AddFinger_RPT","Terminal_Radial_Translate1", "Phalanx",  "Remove_FG", "Terminal_Joint5",  "Terminal_Link3"
#          ]

for rule in rules:
    G.apply_rule(rule_vocabul.get_rule(rule))

func_reward = control_optimizer.create_reward_function(G)
# plot_graph(G)
best_control = []
res = -func_reward(best_control, True)
print()
print(res)

# G = GraphGrammar()
# rules = ["Init", 
#          "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Remove_FG", "Terminal_Link3", "Terminal_Joint5", 
#          "RemoveFinger_N", 
#          "RemoveFinger_R", 
#          "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx",  "Remove_FG", "Terminal_Joint5", "Terminal_Link3", 
#          "RemoveFinger_P",
#          "AddFinger_RPT","Terminal_Radial_Translate1", "Phalanx",  "Remove_FG", "Terminal_Joint5",  "Terminal_Link3"
#          ]


# for rule in rules:
#     G.apply_rule(rule_vocabul.get_rule(rule))

# func_reward = control_optimizer.create_reward_function(G)
# # plot_graph(G)
# best_control = []
# res = -func_reward(best_control, True)
# print()
# print(res)

