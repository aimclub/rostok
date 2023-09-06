from rostok.graph_grammar.node import GraphGrammar
#from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.library.rule_sets.ruleset_old_style_nsc import create_rules


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

def get_one_finger_one_link():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Remove_FG", "Terminal_Radial_Translate1",  "Terminal_Link3", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_one_finger_one_rlink():
    graph = GraphGrammar()
    rules = ["Init",
        "RemoveFinger", 
        "RemoveFinger_N",
        "AddFinger_R", "Remove_FG", "Terminal_Radial_Translate1",  "Terminal_Link3", 
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_three_link_one_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate3", "Phalanx_1", "Phalanx", "Phalanx", "Remove_FG",
        "Terminal_Link3", "Terminal_Link2", "Terminal_Link1",
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_two_same_link_one_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link2", 'Terminal_Joint_1', 'Terminal_Base_Joint_3',
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_three_same_link_one_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link2", "Terminal_Link2",
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_four_same_link_one_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Phalanx","Remove_FG",
        "Terminal_Link2", "Terminal_Link2", "Terminal_Link2","Terminal_Link2",
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_RN", 
        "RemoveFinger_P", 
        "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_two_link_three_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Remove_FG", "Terminal_Link3",  "Terminal_Link3", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_P", 

        "AddFinger_RN", "Terminal_Radial_Translate2", "Terminal_Negative_Translate2", "Terminal_Positive_Turn_0" , "Phalanx",
        "Remove_FG",  "Terminal_Link3",  "Terminal_Link3",
        
        "AddFinger_RP", "Terminal_Radial_Translate2", "Terminal_Positive_Translate2", "Terminal_Negative_Turn_0", "Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link3"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_two_link_three_far_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate3", "Phalanx", "Phalanx", "Remove_FG",
        "Terminal_Link3",  "Terminal_Link3", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "AddFinger_RNT", "Terminal_Radial_Translate3", "Phalanx", "Phalanx",
        "Remove_FG",  "Terminal_Link3",  "Terminal_Link3",
        "RemoveFinger_P", 
        "AddFinger_RPT", "Terminal_Radial_Translate3", "Phalanx", "Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link3"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_three_link_three_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx","Phalanx", "Remove_FG",
        "Terminal_Link2",  "Terminal_Link2", "Terminal_Link2", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx","Phalanx",
        "Remove_FG",  "Terminal_Link2",  "Terminal_Link2","Terminal_Link2",
        "RemoveFinger_P", 
        "AddFinger_RPT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx","Phalanx",
        "Remove_FG", "Terminal_Link2", "Terminal_Link2", "Terminal_Link2"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_three_link_three_finger():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx","Phalanx", "Remove_FG",
        "Terminal_Link2",  "Terminal_Link2", "Terminal_Link2", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx","Phalanx",
        "Remove_FG",  "Terminal_Link2",  "Terminal_Link2","Terminal_Link2",
        "RemoveFinger_P", 
        "AddFinger_RPT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx","Phalanx",
        "Remove_FG", "Terminal_Link2", "Terminal_Link2", "Terminal_Link2"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_three_link_three_finger_scale():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx","Phalanx", "Remove_FG",
        "Terminal_Link3",  "Terminal_Link2", "Terminal_Link1", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx","Phalanx",
        "Remove_FG",  "Terminal_Link3",  "Terminal_Link2","Terminal_Link1",
        "RemoveFinger_P", 
        "AddFinger_RPT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx","Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link2", "Terminal_Link1"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph


def get_three_link_three_finger_scale_dist():
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate3", "Phalanx", "Phalanx","Phalanx", "Remove_FG",
        "Terminal_Link3",  "Terminal_Link2", "Terminal_Link1", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "AddFinger_RNT", "Terminal_Radial_Translate3", "Phalanx", "Phalanx","Phalanx",
        "Remove_FG",  "Terminal_Link3",  "Terminal_Link2","Terminal_Link1",
        "RemoveFinger_P", 
        "AddFinger_RPT", "Terminal_Radial_Translate3", "Phalanx", "Phalanx","Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link2", "Terminal_Link1"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph 