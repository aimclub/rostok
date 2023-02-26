import datetime
from copy import deepcopy
from functools import partial
import pickle
import time
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import \
    CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import \
    GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import (
    MutationStrengthEnum, MutationTypesEnum)
from golem.core.optimisers.genetic.operators.regularization import \
    RegularizationTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from obj_grasp.objects import get_obj_hard_ellipsoid, get_object_to_grasp_sphere
from optmizers_config import get_cfg_graph
from rule_sets.ruleset_old_style_graph import create_rules
from rule_sets.rule_extention_golem_edition import rule_vocab, torque_dict
from rostok.adapters.golem_adapter import (GraphGrammarAdapter,
                                           GraphGrammarFactory)
from rostok.graph_grammar.graph_utils import plot_graph
from rostok.graph_grammar.make_random_graph import make_random_graph
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
import random
from mutation_logik import add_mut, del_mut
from golem.core.optimisers.genetic.operators import crossover

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
    terminal_end = rule_vocab.get_rule("TerminalEndLimb1")
    one_finger.apply_rule(one)
    one_finger.apply_rule(upper)
    
    one_finger.apply_rule(terminal_end)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
    one_finger.apply_rule(upper)
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