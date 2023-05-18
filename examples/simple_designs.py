from rostok.graph_grammar.node import GraphGrammar
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules

def get_palm():
    G = GraphGrammar()
    rules = ["Init", 
         "RemoveFinger",  
         "RemoveFinger_N", 
         "RemoveFinger_R", 
         "RemoveFinger_RN", 
         "RemoveFinger_P",
         "RemoveFinger_RP"
         ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))
    
    return G


def get_two_link_three_finger():
    G = GraphGrammar()
    rules = ["Init", 
            "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Remove_FG", "Terminal_Link3", "Terminal_Joint1", "Terminal_Joint6", "Terminal_Link1",
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint2", "Terminal_Link1", "Terminal_Joint6", "Terminal_Link1",
            "RemoveFinger_P",
            "AddFinger_RPT","Terminal_Radial_Translate1", "Phalanx", "Phalanx", "Remove_FG", "Terminal_Joint2",  "Terminal_Link1", "Terminal_Joint6",  "Terminal_Link1"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))
    
    return G

def get_one_link_three_finger():
    G = GraphGrammar()
    rules = ["Init", 
            "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Remove_FG", "Terminal_Link3", "Terminal_Joint5", 
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "AddFinger_RNT", "Terminal_Radial_Translate1", "Phalanx",  "Remove_FG", "Terminal_Joint5", "Terminal_Link3", 
            "RemoveFinger_P",
            "AddFinger_RPT","Terminal_Radial_Translate1", "Phalanx",  "Remove_FG", "Terminal_Joint5",  "Terminal_Link3"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))
    
    return G

def get_one_link_two_finger():
    G = GraphGrammar()
    rules = ["Init", 
            "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3", 
            "RemoveFinger_N", 
            "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "RemoveFinger_RP"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G

def get_one_link_shifted_one_finger():
    G = GraphGrammar()
    rules = ["Init", 
            "RemoveFinger",  
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "RemoveFinger_P",
            "RemoveFinger_RP"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G

def get_one_link_crossed_finger():
    G = GraphGrammar()
    rules = ["Init", 
            "RemoveFinger",  
            "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "RemoveFinger_R", 
            "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "RemoveFinger_P",
            "RemoveFinger_RP"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G


def get_one_link_four_finger():
    G = GraphGrammar()
    rules = ["Init", 
            "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3", 
            "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "AddFinger_RP","Terminal_Radial_Translate1", "Terminal_Positive_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G

def get_one_link_six_finger():
    G = GraphGrammar()
    rules = ["Init", 
            "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3", 
            "AddFinger_N", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "AddFinger_R", "Terminal_Radial_Translate1", "Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "AddFinger_RN", "Terminal_Radial_Translate1", "Terminal_Negative_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "AddFinger_P","Terminal_Radial_Translate1", "Terminal_Positive_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3",
            "AddFinger_RP","Terminal_Radial_Translate1", "Terminal_Positive_Translate2","Phalanx", "Terminal_Joint5", "Remove_FG", "Terminal_Link3"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G

def get_two_link_one_finger():
    G = GraphGrammar()
    rules = ["Init", 
            "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Remove_FG", "Terminal_Link3", "Terminal_Joint1", "Terminal_Joint6", "Terminal_Link1", 
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "RemoveFinger_RP"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G

def get_three_link_one_finger_with_no_control():
    G = GraphGrammar()
    rules = ["Init", 
            "AddFinger",  "Terminal_Radial_Translate1", "Phalanx", "Phalanx",  "Phalanx", "Remove_FG", "Terminal_Link3", "Terminal_Joint1", "Terminal_Joint6", "Terminal_Link1", "Terminal_Link2", "Terminal_NoControlJoint", 
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "RemoveFinger_RP"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G


def get_one_link_one_finger_double_joint():
    G = GraphGrammar()
    rules = ["Init", 
            "AddFinger",  "Terminal_Radial_Translate1", "Double_Joint_Phalanx", "Remove_FG", "Terminal_Link3", "Terminal_Joint6", "Terminal_Joint2", 
            "RemoveFinger_N", 
            "RemoveFinger_R", 
            "RemoveFinger_RN", 
            "RemoveFinger_P",
            "RemoveFinger_RP"
            ]
    rule_vocabul, _ = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G