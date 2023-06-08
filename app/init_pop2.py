
from rostok.library.rule_sets.rule_extention_merge import rule_vocab 
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.graph_utils import plot_graph

#rule_vocab, _ = create_rules()


def get_non_terminal_one_finger():
    one_finger = GraphGrammar()
    one = rule_vocab.get_rule("InitMechanism_1")
    upper = rule_vocab.get_rule("FingerUpper")
    one_finger.apply_rule(one)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    return one_finger

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

STRUCTURE_ZOO = [get_non_terminal_one_finger, get_non_terminal_two_finger_asym, get_non_terminal_three_finger_short, get_non_terminal_three_finger_long, get_non_terminal_four_finger]
def get_population_zoo():
    zoo = []
    for struct in STRUCTURE_ZOO:
        
        graphs_one_struct = []
        for _ in range(4):
            graph_zoo = struct()
            rule_vocab.make_graph_terminal(graph_zoo)
            graphs_one_struct.append(graph_zoo)
        
        # for _ in range(3):
        #     skoka = random.choice([5, 8, 3, 5, 7])
        #     gr = make_random_graph(skoka, rule_vocab)
        #     graphs_one_struct.append(gr)

        zoo.extend(graphs_one_struct)
    return zoo

def get_pop_simple():
    finger = []
    for _ in range(3):
        nonterminal = get_non_terminal_one_finger()
        rule_vocab.make_graph_terminal(nonterminal)
        finger.append(nonterminal)
    for _ in range(3):
        nonterminal = get_non_terminal_two_finger_asym()
        rule_vocab.make_graph_terminal(nonterminal)
        finger.append(nonterminal)
    return finger