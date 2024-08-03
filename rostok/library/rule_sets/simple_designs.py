from rostok.graph_grammar.node import GraphGrammar
from rostok.library.rule_sets.rulset_simple_fingers import create_rules


def get_palm():
    graph = GraphGrammar()
    rules = [
        "Init", "RemoveFinger", "RemoveFinger_N", "RemoveFinger_R", "RemoveFinger_RN",
        "RemoveFinger_P", "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_three_link_one_finger(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate3",  "Phalanx", "Phalanx", "Remove_FG",
        "Terminal_Link3", "Terminal_Link2", "Terminal_Link1","Terminal_Base_Joint_2", "Terminal_Joint_1", "Terminal_Joint_2",
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

def get_three_link_one_finger_independent(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate3",  "Phalanx", "Phalanx", "Remove_FG",
        "Terminal_Link3", "Terminal_Link2", "Terminal_Link1",
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules(smc = smc, tendon=False)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_two_link_three_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger_R", "Phalanx", "Remove_FG", 
        "Terminal_Radial_Translate1","Terminal_Base_Joint_1", 'Terminal_Link2', 'Terminal_Link2',
        "AddFinger_P", "Terminal_Radial_Translate1", "Phalanx", "Remove_FG",
        "Terminal_Negative_Turn_0",'Terminal_Positive_Translate1', 'Terminal_Link2', 'Terminal_Link2',
        "AddFinger_N", "Terminal_Radial_Translate1", "Phalanx", "Remove_FG", 
        "Terminal_Positive_Turn_0",'Terminal_Negative_Translate1', 'Terminal_Link2', 'Terminal_Link2',
        "RemoveFinger_RP",
        "RemoveFinger_RN",
        "RemoveFinger",
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))
    rule_vocabul.make_graph_terminal(graph)
    return graph

def get_two_link_two_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger_R", "Phalanx", "Remove_FG", 
        "Terminal_Radial_Translate1","Terminal_Base_Joint_1", 'Terminal_Link1', 'Terminal_Link1',
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Remove_FG",
        "Terminal_Base_Joint_1", 'Terminal_Link1', 'Terminal_Link1',
        "RemoveFinger_RP",
        "RemoveFinger_P",
        "RemoveFinger_RN",
        "RemoveFinger_N",
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))
    rule_vocabul.make_graph_terminal(graph)
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
    rule_vocabul, _ = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_one_link_two_finger():
    graph = GraphGrammar()
    rules = ["Init",
            "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5",
            "Remove_FG", "Terminal_Link3", 
            "RemoveFinger_N", 
            "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", 
            "Remove_FG", "Terminal_Link3",
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "RemoveFinger_RP"
    ]
    rule_vocabul, _ = create_rules()
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
    rule_vocabul, _ = create_rules()
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
    rule_vocabul, _ = create_rules()
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
    rule_vocabul, _ = create_rules()
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
    rule_vocabul, _ = create_rules()
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
    rule_vocabul, _ = create_rules()
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
    rule_vocabul, _ = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph