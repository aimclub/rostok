from rostok.graph_grammar.node import GraphGrammar
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.graph_grammar.node import Node
from rostok.block_builder_api.block_blueprints import TransformBlueprint, PrimitiveBodyBlueprint, RevolveJointBlueprint
from rostok.graph_grammar.graph_utils import plot_graph, plot_graph_ids
from rostok.simulation_chrono.basic_simulation import SystemPreviewChrono
from rostok.graph_grammar.mutation import add_node_between

def get_palm():
    graph = GraphGrammar()
    rules = [
        "Init", "RemoveFinger", "RemoveFinger_N", "RemoveFinger_R", "RemoveFinger_RN",
        "RemoveFinger_P", "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_two_link_three_finger_same():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG",
        "Terminal_Link3", "Terminal_Joint1", "Terminal_Joint6", "Terminal_Link3", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx",
        "Remove_FG", "Terminal_Joint2", "Terminal_Link3", "Terminal_Joint6", "Terminal_Link3",
        "RemoveFinger_P", 
        "AddFinger_RPT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx",
        "Remove_FG", "Terminal_Joint2", "Terminal_Link3", "Terminal_Joint6", "Terminal_Link3"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_one_link_three_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Remove_FG", "Terminal_Link3",
            "Terminal_Joint5", 
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "AddFinger_RNT","Terminal_Radial_Translate1", "Phalanx", "Remove_FG", 
            "Terminal_Joint5", "Terminal_Link3",
            "RemoveFinger_P", 
            "AddFinger_RPT", "Terminal_Radial_Translate1", "Phalanx", "Remove_FG",
            "Terminal_Joint5", "Terminal_Link3"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_one_link_two_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5",
            "Remove_FG", "Terminal_Link3", 
            
            "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", 
            "Remove_FG", "Terminal_Link3",
            "RemoveFinger_N", 
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_one_link_shifted_one_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "RemoveFinger", 
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2", "Phalanx", 
            "Terminal_Joint5", "Remove_FG", "Terminal_Link3", 
            "RemoveFinger_P", 
            "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_one_link_crossed_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "RemoveFinger", 
            "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2", "Phalanx", 
            "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "RemoveFinger_R", 
            "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2", "Phalanx", 
            "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "RemoveFinger_P", 
            "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_one_link_four_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5",
            "Remove_FG", "Terminal_Link3", 
            "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2", "Phalanx", 
            "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", 
            "Remove_FG", "Terminal_Link3", 
            "RemoveFinger_RN", 
            "RemoveFinger_P", 
            "AddFinger_RP", "Terminal_Radial_Translate1", "Terminal_Positive_Translate2", "Phalanx", 
            "Terminal_Joint5", "Remove_FG", "Terminal_Link3"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_one_link_six_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5",
            "Remove_FG", "Terminal_Link3", 
            "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2", 
            "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", 
            "Remove_FG", "Terminal_Link3", 
            "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2", 
            "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "AddFinger_P", "Terminal_Radial_Translate1", "Terminal_Positive_Translate2", "Phalanx",
            "Terminal_Joint5", "Remove_FG", "Terminal_Link3", 
            "AddFinger_RP", "Terminal_Radial_Translate1", "Terminal_Positive_Translate2", "Phalanx", 
            "Terminal_Joint5", "Remove_FG", "Terminal_Link3"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_two_link_one_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG",
            "Terminal_Link3", "Terminal_Joint1", "Terminal_Joint6", "Terminal_Link1", 
            "RemoveFinger_N",
            "RemoveFinger_R", 
            "RemoveFinger_RN", 
            "RemoveFinger_P", 
            "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_one_link_one_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Remove_FG",
            "Terminal_Link3", "Terminal_Joint6", 
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_two_link_one_finger_transform_0():
    graph = get_two_link_one_finger()
    tbp1 = TransformBlueprint()
    zero_trans_node = Node("T0", True, tbp1)
    add_node_between((17, 19), graph, zero_trans_node)
    return graph

def get_2_3_link_2_finger():
    graph = GraphGrammar()
    rules = ["Init",
            
            "AddFinger", "Terminal_Radial_Translate1", 
            "Phalanx", "Terminal_Joint5", "Terminal_Link1",
            "Phalanx", "Terminal_Joint5", "Terminal_Link1", 
            "Phalanx", "Terminal_Joint5", "Terminal_Link2", 
            "Remove_FG", 
            
            "AddFinger_R", "Terminal_Radial_Translate1", 
            "Phalanx", "Terminal_Joint5", "Terminal_Link1", 
            "Phalanx", "Terminal_Joint5", "Terminal_Link2", "Remove_FG",
            
            "RemoveFinger_N", 
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_5_link_one_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "AddFinger", "Terminal_Radial_Translate1", 
            "Phalanx", "Terminal_Link2", "Terminal_Joint6",
            "Phalanx", "Terminal_Link2", "Terminal_Joint6",
            "Phalanx", "Terminal_Link2", "Terminal_Joint6",
            "Phalanx", "Terminal_Link2", "Terminal_Joint6",
            "Phalanx", "Terminal_Link2", "Terminal_Joint6",
            "Remove_FG", 
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules(True)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

if __name__ == "__main__":
    graph = get_5_link_one_finger()
    plot_graph(graph)
    plot_graph_ids(graph)
    sim = SystemPreviewChrono()
    sim.add_design(graph)
    sim.simulate(100000, camera_pos=(-0.51, 0.51, -0.51))