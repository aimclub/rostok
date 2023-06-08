
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.graph_utils import plot_graph
from rostok.simulation_chrono import basic_simulation
from rostok.graph_grammar import rule_vocabulary
rule_vocab, _ = create_rules()
from random import choices

def get_non_terminal_one_finger():
    mister_finger = GraphGrammar()
    one = rule_vocab.get_rule("Init")
    add_fingers = [rule_vocab.get_rule("AddFinger"), rule_vocab.get_rule("AddFinger_R"), rule_vocab.get_rule("AddFinger_P"),
            rule_vocab.get_rule("AddFinger_PT"), rule_vocab.get_rule("AddFinger_N"), rule_vocab.get_rule("AddFinger_NT"),
            rule_vocab.get_rule("AddFinger_RP"), rule_vocab.get_rule("AddFinger_RPT"), rule_vocab.get_rule("AddFinger_RN"),
            rule_vocab.get_rule("AddFinger_RNT")]
    number_deleted_finger = 3
    remove_finger = ["RemoveFinger", "RemoveFinger_R", "Remove_FG",
        "RemoveFinger_P", "RemoveFinger_N", "RemoveFinger_RP",
        "RemoveFinger_RN"]
    
    deleted_finger_rule_name  = choices(remove_finger, k=number_deleted_finger)
    mister_finger.apply_rule(one)
    for i in deleted_finger_rule_name:
        rule_del = rule_vocab.get_rule(i)
        mister_finger.apply_rule(rule_del)
    


 
    return mister_finger

def get_non_terminal_two_finger_asym():
    one_finger = GraphGrammar()
    one = rule_vocab.get_rule("InitMechanism_2")
    upper = rule_vocab.get_rule("FingerUpper")
    terminal_end = rule_vocab.get_rule("Remove_EM")
    one_finger.apply_rule(one)
    one_finger.apply_rule(upper)
    

    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(terminal_end)
    return one_finger

def get_non_terminal_three_finger_short():
    one_finger = GraphGrammar()
    one = rule_vocab.get_rule("InitMechanism_3_R_A")
    upper = rule_vocab.get_rule("FingerUpper")
    one_finger.apply_rule(one)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    return one_finger

def get_non_terminal_three_finger_long():
    one_finger = GraphGrammar()
    one = rule_vocab.get_rule("InitMechanism_3_L")
    upper = rule_vocab.get_rule("FingerUpper")
    one_finger.apply_rule(one)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    return one_finger

def get_non_terminal_four_finger():
    one_finger = GraphGrammar()
    one = rule_vocab.get_rule("InitMechanism_4_A")
    upper = rule_vocab.get_rule("FingerUpper")
    one_finger.apply_rule(one)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    return one_finger


graph_non_terminal_one_finger = get_non_terminal_one_finger()
prew = basic_simulation.SystemPreviewChrono()
prew.add_design(graph_non_terminal_one_finger)
prew.simulate(1000)
