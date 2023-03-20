from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import mcts
# imports from standard libs
import networkx as nx
import optmizers_config
# chrono imports
import pychrono as chrono
import simple_designs

from rostok.graph_generators.mcts_helper import (make_mcts_step,
                                                 prepare_mcts_state_and_helper)
from rostok.graph_grammar.graph_grammar import GraphGrammar
from rostok.library.obj_grasp.objects import (get_obj_easy_box,
                                              get_obj_easy_large_box,
                                              get_obj_easy_long_box,
                                              get_obj_hard_ellipsoid,
                                              get_obj_hard_large_ellipsoid,
                                              get_obj_hard_long_ellipsoid,
                                              get_object_to_grasp_sphere)
from rostok.library.rule_sets import rule_extention_graph
from rostok.library.rule_sets.ruleset_old_style_graph_nonails import \
    create_rules
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from rostok.virtual_experiment_chrono.flags_simualtions import (FlagMaxTime,
                                                                FlagNotContact,
                                                                FlagSlipout)


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=500,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.savefig("three_finger.svg")

def simulate_rules(_rules, _control_optimizer):
    _graph = GraphGrammar()
    for _rule in _rules:
        _graph.apply_rule(rule_vocabul.get_rule(_rule))

    func_reward = control_optimizer.create_reward_function(_graph)
    res = -func_reward([], True)
    print()
    print(res)
# %% Create extension rule vocabulary
rule_vocabul, torque_dict = create_rules()
#rule_vocabul = deepcopy(rule_extention_graph.rule_vocab)
#torque_dict = rule_extention_graph.torque_dict

cfg = optmizers_config.get_cfg_graph(torque_dict)
# cfg.gravity_vector = [0, -9.8, 0]
# cfg.time_saturation_gravity = 0.5
# cfg.time_start_gravity = 0.1
cfg.get_rgab_object_callback = get_obj_easy_large_box
control_optimizer = ControlOptimizer(cfg)

graph = simple_designs.get_two_link_three_finger()
func_reward = control_optimizer.create_reward_function(graph)
#plot_graph(graph)
res = -func_reward([], True)
print()
print(res)

# rules = ["Init", "AddFinger_N", "AddFinger",  "Terminal_Negative_Translate2",  "RemoveFinger_P",  "Terminal_Radial_Translate1", "RemoveFinger_RN", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Radial_Translate1", "Remove_FG", "Terminal_Link3", "AddFinger_R", "RemoveFinger_RP", "Phalanx", "Terminal_Joint5", "Terminal_Link3", "Remove_FG", "Terminal_Radial_Translate1"]
# simulate_rules(rules, control_optimizer)

# graph = simple_designs.get_one_link_two_finger()
# func_reward = control_optimizer.create_reward_function(graph)
# res = -func_reward([], True)
# print()
# print(res)

# rules = ["Init", 
#          "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Remove_FG", "Terminal_Link3", "Terminal_Joint5", "Terminal_Link2", "Terminal_Joint2",
#          "AddFinger_NT", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint3", "Remove_FG", "Terminal_Link3",
#          "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint3", "Remove_FG", "Terminal_Link3",
#          "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5", "Terminal_Link3", "Terminal_Joint2", "Terminal_Link2",
#          "AddFinger_PT", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint3", "Remove_FG", "Terminal_Link3",
#          "AddFinger_RPT","Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5",  "Terminal_Link3", "Terminal_Joint2",  "Terminal_Link2"
#          ]
# simulate_rules(rules, control_optimizer)

# rules = ["Init", 
#          "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Remove_FG", "Terminal_Link3", "Terminal_Joint5", "Terminal_Link2", "Terminal_Joint2",
#          "RemoveFinger_N", 
#          "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint3", "Remove_FG", "Terminal_Link3",
#          "RemoveFinger_RN", 
#          "RemoveFinger_P",
#          "RemoveFinger_RP"
#          ]

# simulate_rules(rules, control_optimizer)

# rules = ["Init", 
#         "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Remove_FG", "Terminal_Link3", "Terminal_Joint5", "Terminal_Link2", "Terminal_Joint2",
#         "RemoveFinger_N", 
#         "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint3", "Remove_FG", "Terminal_Link3",
#         "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5", "Terminal_Link3", "Terminal_Joint2", "Terminal_Link2",
#         "RemoveFinger_P",
#         "AddFinger_RPT","Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint5",  "Terminal_Link3", "Terminal_Joint2",  "Terminal_Link2"
#         ]

# simulate_rules(rules, control_optimizer)