from rostok.graph_grammar.node import GraphGrammar
from rostok.library.rule_sets.ruleset_simple_fingers import create_rules

def get_palm(smc = False):
    graph = GraphGrammar()
    rules = [
        "Init", "RemoveFinger", "RemoveFinger_N", "RemoveFinger_R", "RemoveFinger_RN",
        "RemoveFinger_P", "RemoveFinger_RP"
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_one_finger_one_link(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Remove_FG", "Terminal_Radial_Translate1",  "Terminal_Link3", "Terminal_Base_Joint_2",
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

def get_one_finger_one_rlink(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "RemoveFinger", 
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

def get_two_same_link_one_finger(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link2", 'Terminal_Joint_1', 'Terminal_Base_Joint_3', "Terminal_Base_Joint_2", "Terminal_Joint_2",
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

def get_three_link_one_finger(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate3",  "Phalanx", "Phalanx", "Remove_FG",
        "Terminal_Link3", "Terminal_Link2", "Terminal_Link1","Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2",
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

def get_three_same_link_one_finger(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG",
        "Terminal_Link2", "Terminal_Link2", "Terminal_Link2", "Terminal_Base_Joint_3", "Terminal_Joint_1", "Terminal_Joint_1",
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

def get_four_same_link_one_finger(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Phalanx","Remove_FG",
        "Terminal_Link2", "Terminal_Link2", "Terminal_Link2","Terminal_Link2", "Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2",  "Terminal_Joint_2",
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

def get_two_link_three_finger(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate1", "Phalanx", "Remove_FG", "Terminal_Link3",  "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_P", 

        "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2", "Terminal_Positive_Turn_0" , "Phalanx",
        "Remove_FG",  "Terminal_Link3",  "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", 
        
        "AddFinger_RP", "Terminal_Radial_Translate1", "Terminal_Positive_Translate2", "Terminal_Negative_Turn_0", "Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", 
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_two_link_three_finger_rotated(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Remove_FG", "Terminal_Link3",  "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_P", 

        "AddFinger_RN", "Terminal_Radial_Translate2", "Terminal_Negative_Translate2", "Terminal_Positive_Turn_1" , "Phalanx",
        "Remove_FG",  "Terminal_Link3",  "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", 
        
        "AddFinger_RP", "Terminal_Radial_Translate2", "Terminal_Positive_Translate2", "Terminal_Negative_Turn_1", "Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", 
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph

def get_three_link_three_finger(smc = False):
    graph = GraphGrammar()
    rules = ["Init",
        "AddFinger", "Terminal_Radial_Translate2", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Link3", "Terminal_Link3", "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2", 
        "RemoveFinger_N",
        "RemoveFinger_R", 
        "RemoveFinger_P", 

        "AddFinger_RN", "Terminal_Radial_Translate2", "Terminal_Negative_Translate2", "Terminal_Positive_Turn_0" , "Phalanx", "Phalanx",
        "Remove_FG",  "Terminal_Link3",  "Terminal_Link3", "Terminal_Link3","Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2",
        
        "AddFinger_RP", "Terminal_Radial_Translate2", "Terminal_Positive_Translate2", "Terminal_Negative_Turn_0", "Phalanx", "Phalanx",
        "Remove_FG", "Terminal_Link3", "Terminal_Link3", "Terminal_Link3", "Terminal_Base_Joint_2", "Terminal_Joint_2", "Terminal_Joint_2", 
    ]
    rule_vocabul = create_rules(smc = smc)
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph
