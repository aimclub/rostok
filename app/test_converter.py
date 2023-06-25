import numpy as np

from rostok.graph_generators.graph_heuristic_search.torch_adapter import TorchAdapter
from rostok.graph_generators.graph_heuristic_search.design_environment import DesignEnvironment
from rostok.graph_generators.graph_heuristic_search.random_search import RandomSearch
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.graph_grammar.node import GraphGrammar
import networkx as nx

from mcts_run_setup import config_with_standard_graph
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
from rostok.library.rule_sets.ruleset_new_style_graph import create_rules

rule_vocabul, torque_dict = create_rules()

# create blueprint for object to grasp
grasp_object_blueprint = get_object_parametrized_sphere(0.4, 0.7)

# create reward counter using run setup function
control_optimizer = config_with_standard_graph(grasp_object_blueprint, torque_dict)

# def get_palm():
#     graph = GraphGrammar()
#     rules = [
#         "Init", "RemoveFinger", "RemoveFinger_N", "RemoveFinger_R", "RemoveFinger_RN",
#         "RemoveFinger_P", "RemoveFinger_RP"
#     ]
#     rule_vocabul, _ = create_rules()
#     for rule in rules:
#         graph.apply_rule(rule_vocabul.get_rule(rule))

#     return graph, rule_vocabul

# def get_two_link_three_finger():
#     graph = GraphGrammar()
#     rules = ["Init",
#         "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG",
#         "Terminal_Link3", "Terminal_Joint1", "Terminal_Joint6", "Terminal_Link1", 
#         "RemoveFinger_N",
#         "RemoveFinger_R", 
#         "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx",
#         "Remove_FG", "Terminal_Joint2", "Terminal_Link1", "Terminal_Joint6", "Terminal_Link1",
#         "RemoveFinger_P", 
#         "AddFinger_RPT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx",
#         "Remove_FG", "Terminal_Joint2", "Terminal_Link1", "Terminal_Joint6", "Terminal_Link1"
#     ]
#     rule_vocabul, _ = create_rules()
#     for rule in rules:
#         graph.apply_rule(rule_vocabul.get_rule(rule))

#     return graph, rule_vocabul

# palm, rule_vocab_p = get_palm()
# mechs_2l_3f, rule_vocab_2l_3f = get_two_link_three_finger()

# coverter = TorchAdapter(rule_vocab_2l_3f.node_vocab)

design_env = DesignEnvironment(rule_vocabul, control_optimizer)

design_env.load_environment("./rostok/graph_generators/graph_heuristic_search/dataset_design_space/rnd_srch_11h43m_date_25d6m2023y")
design_env.load_environment("./rostok/graph_generators/graph_heuristic_search/dataset_design_space/rnd_srch_2h54m_date_25d6m2023y")
rnd_srch = RandomSearch(15)
rnd_srch.search(design_env, 10000)

None