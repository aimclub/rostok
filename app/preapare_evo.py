from rostok.graph_grammar.graphgrammar_explorer import random_search_mechs_n_branch

from rostok.graph_grammar.node import GraphGrammar, Node
from rostok.graph_grammar.mutation import add_mut, del_mut
from rostok.library.rule_sets import ruleset_old_style_graph
from rostok.block_builder_chrono.blocks_utils import FrameTransform
from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.graph_grammar.node import ROOT
from rostok.graph_grammar import rule_vocabulary
from rostok.block_builder_api.block_blueprints import TransformBlueprint, PrimitiveBodyBlueprint, RevolveJointBlueprint
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_api.block_parameters import JointInputType
import pickle

rule_vocab, _ = ruleset_old_style_graph.create_rules()
terminal_nodes_dict = rule_vocab.node_vocab.terminal_node_dict
terminal_nodes_dict.pop("FT")

RX_b = TransformBlueprint(FrameTransform([0, 0, 0], [0.707, 0.707, 0, 0]))
RY_b = TransformBlueprint(FrameTransform([0, 0, 0], [0.707, 0, 0.707, 0]))
RZ_b = TransformBlueprint(FrameTransform([0, 0, 0], [0.707, 0, 0, 0.707]))

x2_b = TransformBlueprint(FrameTransform([0.2, 0, 0], [1, 0, 0, 0]))
y2_b = TransformBlueprint(FrameTransform([0, 0.2, 0], [1, 0, 0, 0]))
z2_b = TransformBlueprint(FrameTransform([0, 0, 0.2], [1, 0, 0, 0]))

node_rx = Node("RX", True, RX_b)
node_ry = Node("RY", True, RY_b)
node_rz = Node("RZ", True, RZ_b)

node_x = Node("RX", True, x2_b)
node_y = Node("RY", True, y2_b)
node_z = Node("RZ", True, z2_b)

#terminal_nodes_dict
terminal_nodes = terminal_nodes_dict.values()
terminal_nodes_tr = [node_rx, node_ry, node_rz, node_x, node_y, node_z]

def create_balance_population(rule_vocabul, max_tries=10000):

    two_fingers = random_search_mechs_n_branch(rule_vocabul,
                                               category_size=8,
                                               numbers_of_rules=[6, 8],
                                               desired_branch=2,
                                               max_tries=max_tries)
    three_fingers = random_search_mechs_n_branch(rule_vocabul,
                                                 category_size=8,
                                                 numbers_of_rules=[8, 10],
                                                 desired_branch=3,
                                                 max_tries=max_tries)
    four_fingers = random_search_mechs_n_branch(rule_vocabul,
                                                category_size=8,
                                                numbers_of_rules=[10, 11],
                                                desired_branch=4,
                                                max_tries=max_tries)

    return two_fingers + three_fingers + four_fingers


def custom_mutation_add_body(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    graph_mut = add_mut(graph, terminal_nodes, (1, 0, 0))
    return graph_mut

def custom_mutation_add_j(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    graph_mut = add_mut(graph, terminal_nodes, (0, 0, 1))
    return graph_mut

def custom_mutation_add_tr(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    graph_mut = add_mut(graph, terminal_nodes_tr, (0, 1, 0))
    return graph_mut

def custom_mutation_del_body(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    graph_mut = del_mut(graph, terminal_nodes, (1, 0, 0))
    return graph_mut

def custom_mutation_del_tr(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    graph_mut = del_mut(graph, terminal_nodes, (0, 1, 0))
    return graph_mut

def custom_mutation_del_j(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    graph_mut = del_mut(graph, terminal_nodes, (0, 0, 1))
    return graph_mut

def load_init_population() -> list[GraphGrammar]:
    try:
        with open('init_population_cache.pickle', 'rb') as f:
            data_new = pickle.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "init_population_cache.pickle. Please generate cache population by using preapare_evo.py"
        ) from exc

    return data_new


if __name__ == "__main__":
    init_population = create_balance_population(rule_vocab)
    with open('init_population_cache.pickle', 'wb') as f:
        pickle.dump(init_population, f)
