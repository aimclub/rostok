from copy import deepcopy
from functools import partial
from itertools import product
import random
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.simulation_chrono.simulation import (ChronoSystems, EnvCreator, SingleRobotSimulation,
                                                 ChronoVisManager)
from test_ruleset import (get_terminal_graph_three_finger, get_terminal_graph_two_finger,
                          get_terminal_graph_two_finger_mix, rule_vocab)
from rostok.graph_grammar.mutation import add_mut, del_mut
from rostok.graph_grammar.crossovers import subtree_crossover
from rostok.graph_grammar import make_random_graph


def add_mutation_rule(graph: GraphGrammar, rule_vocab1: RuleVocabulary):
    return add_mut(graph, rule_vocab1.node_vocab.terminal_node_dict.values())


def del_mutation_rule(graph: GraphGrammar, rule_vocab1: RuleVocabulary):
    return del_mut(graph, rule_vocab1.node_vocab.terminal_node_dict.values())


def mock_build_mech(graph: GraphGrammar):
    """Build graph and do 2 simulation step

    Args:
        graph (GraphGrammar):

    Returns:
        Penalty: Random positive value [0:5]
    """
    # Build graph
    system = ChronoSystems.chrono_NSC_system(gravity_list=[0, -10, 0])
    env_creator = EnvCreator([])
    vis_manager = ChronoVisManager(0.01)
    sim = SingleRobotSimulation(system, env_creator, vis_manager)
    sim.add_design(graph)
    sim.simulate(2, False)


def create_random_mechs(number):
    mechs = []
    num_rules = [4, 5, 8]
    for _ in range(number):
        num_rule = random.choice(num_rules)
        graph = make_random_graph.make_random_graph(num_rule, rule_vocab)
        mechs.append(graph)
    return mechs

def test_build_crossovered_mechs():
    mechs = create_random_mechs(16)
    combination = list(product(mechs, mechs))
    tets_mechs = []
    for graph_1, graph_2 in combination:
        graph_cross_1, graph_cross_2 = subtree_crossover(graph_1, graph_2)
        tets_mechs.extend([graph_cross_1, graph_cross_2])
    
    test_mechs_and_source = zip(tets_mechs, combination)
    for mech, source_pair in test_mechs_and_source:
        mock_build_mech(mech)


def test_build_mutation_mechs():
    mechs = create_random_mechs(10)
    number_mutation = [2, 5, 4]
    tets_mechs = []
    del_mut = partial(del_mutation_rule, rule_vocab1=rule_vocab)
    add_mut = partial(add_mutation_rule, rule_vocab1=rule_vocab)

    for mech in mechs:
        muted_graph = deepcopy(mech)
        num_mut = random.choice(number_mutation)
        for i in range(num_mut):
            current_mut = random.choice([del_mut, add_mut])
            muted_graph = current_mut(muted_graph)
        tets_mechs.append(muted_graph)
    
    for mech in tets_mechs:
        mock_build_mech(mech)
