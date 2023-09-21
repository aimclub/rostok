from rostok.graph_grammar.node import GraphGrammar
from rostok.library.rule_sets.ruleset_simple_fingers import create_rules

import hyperparameters as hp
import matplotlib.pyplot as plt

import numpy as np
from mcts_run_setup import config_with_tendon
from rostok.library.obj_grasp.objects import (get_object_ellipsoid_h)
from rostok.graph_grammar.graph_utils import plot_graph

grasp_object_blueprint = get_object_ellipsoid_h(0.001, 0.001, 0.001, 0, 10)
control_optimizer = config_with_tendon(grasp_object_blueprint)
simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_scenario

# first generation

def get_free_palm(smc = False):
    graph = GraphGrammar()
    rules = [
        "Init"
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def mech_1(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "RemoveFinger", 
        "RemoveFinger_N",
        "AddFinger_RN", "Terminal_Radial_Translate2", "Terminal_Negative_Translate2", "Terminal_Positive_Turn_0" , "Phalanx",
        "Remove_FG",  "Terminal_Link3",  "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", 
        "RemoveFinger_R", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def mech_2(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link3", 'Terminal_Joint_2', 'Terminal_Base_Joint_2',
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def mech_3(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "RemoveFinger", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "AddFinger_RP", "Terminal_Radial_Translate3", "Terminal_Positive_Translate3", "Terminal_Negative_Turn_2", "Phalanx","Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link2", "Terminal_Link1", "Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2", 
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def mech_4(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link3", 'Terminal_Joint_2', 'Terminal_Base_Joint_2',
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "AddFinger_RN", "Terminal_Radial_Translate3", "Terminal_Negative_Translate2", "Terminal_Positive_Turn_0" , "Phalanx", "Phalanx",
        "Remove_FG",  "Terminal_Link1",  "Terminal_Link3",  "Terminal_Link2","Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def mech_5(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link3", 'Terminal_Joint_2', 'Terminal_Base_Joint_2',
        "RemoveFinger_N",
        "AddFinger_R", "Remove_FG", "Terminal_Radial_Translate1",  "Terminal_Link3", "Terminal_Base_Joint_2",
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def mech_6(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link3", 'Terminal_Joint_2', 'Terminal_Base_Joint_2',
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_RN", 
        "AddFinger_P", "Terminal_Radial_Translate2", "Terminal_Positive_Translate2", "Terminal_Negative_Turn_1", "Phalanx","Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link2", "Terminal_Link1", "Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def mech_7(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link3", 'Terminal_Joint_2', 'Terminal_Base_Joint_2',
        "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate3", "Terminal_Positive_Turn_2", "Phalanx", "Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link3", "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2",
        "RemoveFinger_R", 
        "AddFinger_RN", "Terminal_Radial_Translate3", "Terminal_Negative_Translate2", "Terminal_Positive_Turn_0" , "Phalanx", "Phalanx",
        "Remove_FG",  "Terminal_Link1",  "Terminal_Link3",  "Terminal_Link2","Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2", 
        "RemoveFinger_P",
        "RemoveFinger_RP",  

    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def mech_8(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link3", 'Terminal_Joint_2', 'Terminal_Base_Joint_2',
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "AddFinger_RN", "Terminal_Radial_Translate3", "Terminal_Negative_Translate2", "Terminal_Positive_Turn_0" , "Phalanx", "Phalanx",
        "Remove_FG",  "Terminal_Link1",  "Terminal_Link3",  "Terminal_Link2","Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2", 
        "RemoveFinger_RP", 
        "AddFinger_P", "Terminal_Radial_Translate2", "Terminal_Positive_Translate2", "Terminal_Negative_Turn_0", "Phalanx", "Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link3", "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2",
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def mech_9(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link3", 'Terminal_Joint_2', 'Terminal_Base_Joint_2',
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "AddFinger_RN", "Terminal_Radial_Translate3", "Terminal_Negative_Translate2", "Terminal_Positive_Turn_0" , "Phalanx", "Phalanx",
        "Remove_FG",  "Terminal_Link1",  "Terminal_Link3",  "Terminal_Link2","Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2", 
        "RemoveFinger_P", 
        "AddFinger_RP", "Terminal_Radial_Translate2", "Terminal_Positive_Translate2", "Terminal_Negative_Turn_0", "Phalanx", "Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link3", "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2",
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


graphs = [(mech_1(), [10], [[-25, 0]]), (mech_2(), [10], [[-25, 0]]), (mech_3(), [15], [[-25, 0,0]]), 
          (mech_4(), [10, 15], [[-25, 0], [-25, 0, 0]]), (mech_5(), [5, 10], [[-25], [-25, 0] ]), (mech_6(), [10, 15], [[-25, 0], [-25, 0,0]]),
          (mech_7(), [10, 15, 15], [[-25, 0], [-25, 0,0], [-25, 0,0]]), (mech_8(), [10, 15, 15], [[-25, 0], [-25, 0,0], [-25, 0,0]]), (mech_9(), [10, 15, 15], [[-25, 0], [-25, 0,0], [-25, 0,0]])]


for graph_data in graphs:
    graph = graph_data[0]
    control = graph_data[1]
    angles = graph_data[2]
    data = control_optimizer.optim_parameters2data_control(control, graph)
    simulation_output = simulation_manager.run_simulation(graph, data, angles, True, False)


grasp_object_blueprint = get_object_ellipsoid_h(0.1, 0.12, 0.2, 0, 0.15)
control_optimizer = config_with_tendon(grasp_object_blueprint)
simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_scenario
graph = mech_9()
control = [10, 15, 15]
angles = [[-25.0, 0, 0], [-25, 0, 0], [-25, 0, 0]]
data = control_optimizer.optim_parameters2data_control(control, graph)
simulation_output = simulation_manager.run_simulation(graph, data, angles, True, True)

# uncomment for graphs, but dont forget to comment visualization  
# plot_graph(get_free_palm())
# plot_graph(mech_2())
# plot_graph(mech_4())
# plot_graph(mech_9())