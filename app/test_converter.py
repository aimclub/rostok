import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils

from rostok.graph_generators.graph_heuristic_search.torch_adapter import TorchAdapter
from rostok.neural_network.wrapper import NNWraper 
from rostok.neural_network.old_sagpool import SAGPoolToAlphaZero
from rostok.neural_network.robogrammar_net import Net
from rostok.graph_generators.graph_heuristic_search.design_environment import DesignEnvironment
from rostok.graph_generators.graph_heuristic_search.random_search import RandomSearch
from rostok.graph_generators.graph_heuristic_search.graph_heuristic_search import GraphHeuristicSearch
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules


from mcts_run_setup import config_with_standard_graph
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


args = {
    "device":torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "max_nonterminal_rules":15,
    "max_nonterminal_actions":15,
    "batch_size":64,
    "start_eps":1,
    "end_eps":0.01,
    "eps_decay":0.3,
    "num_designs":20,
    "eps_design":0.3,
    "minibatch":64,
    "opt_iter":50,
    "max_nodes": 20,
    "nhid":512,
    "pooling_ratio":0.6,
    "dropout_ratio":0.5
}

design_env = DesignEnvironment(rule_vocabul, control_optimizer)

coverter = TorchAdapter(rule_vocabul.node_vocab)

args["num_features"] = len(design_env.node2id)
# nnet = SAGPoolToAlphaZero(args)
nnet = Net(args)
nn_wrapper = NNWraper(nnet, args)

ghs = GraphHeuristicSearch(coverter, nn_wrapper, args)
# rnd_srch = RandomSearch(15)
# rnd_srch.search(design_env, 10000)

ghs.search(10, design_env)
None