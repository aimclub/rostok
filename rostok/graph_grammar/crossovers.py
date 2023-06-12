from copy import deepcopy
import random

import rostok.graph_grammar.node as rostok_graph
from rostok.graph_grammar.node import Node
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import networkx as nx
from typing import Callable, List, Optional
from functools import partial
from rostok.graph_grammar.node_block_typing import NodeFeatures

CALLBACK_NEIGHBORS_TYPE = Callable[[List[Node], List[Node], Node], bool]
""" If node is available for replacement, returns true
Args: 
    1 List[Node]: in nodes (predecessors)
    2 List[Node]: out nodes (successors) 
    3 Node: Replacer  
"""


def available_node_ids_based_on_neighbors(replacer_node: Node, graph: rostok_graph.GraphGrammar,
                                          check_neighbors: CALLBACK_NEIGHBORS_TYPE):
    node_ids = list(graph.nodes)
    successors = [graph.successors(id_n) for id_n in node_ids]
    predecessors = [graph.predecessors(id_n) for id_n in node_ids]
    package = zip(node_ids, predecessors, successors)
    passed_ids = [
        node_ids_i for node_ids_i, predecessors_i, successors_i in package
        if check_neighbors(predecessors_i, successors_i, replacer_node)
    ]
    return passed_ids


def available_node_ids_both_directions(graph_1: rostok_graph.GraphGrammar, id_1: int,
                                       graph_2: rostok_graph.GraphGrammar,
                                       check_neighbors: CALLBACK_NEIGHBORS_TYPE):
    replacer = graph_1.get_node_by_id(id_1)
    successors_1 = graph_1.successors(id_1)
    predecessors_1 = graph_1.predecessors(id_1)
    check_neighbors_replacer = partial(check_neighbors, predecessors_1, successors_1)
    available_ids_2 = available_node_ids_based_on_neighbors(replacer, graph_2, check_neighbors)

    available_ids_replacer_2 = [
        id_i for id_i in available_ids_2 if check_neighbors_replacer(graph_2.get_node_by_id(id_i))
    ]
    return available_ids_replacer_2


def is_body(obj: Optional[Node]):
    if obj is list:
        return False
    return NodeFeatures.is_body(obj)


def is_joint(obj: Optional[Node]):
    if obj is list:
        return False
    return NodeFeatures.is_joint(obj)


def is_transform(obj: Optional[Node]):
    if obj is list:
        return False
    return NodeFeatures.is_transform(obj)


avalibale_node_type_p = [is_body, is_joint, is_transform]

any_from = lambda x: any([i(x) for i in avalibale_node_type_p])


def checker(predecessors: List[Node], successors: List[Node]):
    if len(predecessors) == 0:
        is_p = False
    else:
        is_p = any(map(any_from, predecessors))

    if len(successors) == 0:
        is_s = False
    else:
        is_s = any(map(any_from, successors))
    return is_p and is_s


def callback_body(predecessors: List[Node], successors: List[Node], replacer: Node) -> bool:
    if NodeFeatures.is_body(replacer):
        return True
    else:
        return False


def get_subtree_graph(graph: rostok_graph.GraphGrammar, id: int):
    digraph_nx: nx.DiGraph = dfs_tree(graph, id)

    subtree = dict(digraph_nx.nodes)
    nodes_data = dict(graph.nodes(data=True))

    keys_in_subtree = subtree.keys() & nodes_data.keys()
    nodes_data = {key: nodes_data[key] for key in keys_in_subtree}
    subtree.update(nodes_data)
    # Update data nodes in graph
    digraph_nx._node = subtree
    return digraph_nx


def remove_subtree_without_root(graph_1: rostok_graph.GraphGrammar, id_1: int):
    graph_1 = deepcopy(graph_1)
    subtree_to_remove = dfs_tree(graph_1, id_1)
    subtree_to_remove.remove_node(id_1)
    graph_1.remove_edges_from(subtree_to_remove.edges)
    graph_1.remove_nodes_from(subtree_to_remove.nodes)

    return graph_1


def add_subtree_using_rule(graph_1: rostok_graph.GraphGrammar, id_1: int, subtree: nx.DiGraph):
    graph_1 = deepcopy(graph_1)
    rule = rostok_graph.Rule()
    rule.graph_insert = subtree
    # Normal result is list with one element
    root_id_list = [i[0] for i in list(subtree.in_degree) if i[1] == 0]
    if len(root_id_list) > 1:
        raise Exception("Subtree has multi root")
    elif not root_id_list:
        raise Exception("Subtree has no roots")
    subtree_root = root_id_list[0]
    rule.id_node_connect_child = subtree_root
    rule.id_node_connect_parent = subtree_root
    graph_1._replace_node(id_1, rule)
    return graph_1


def subtree_crossover_select(graph_1: rostok_graph.GraphGrammar, graph_2: rostok_graph.GraphGrammar,
                             id_1: int, id_2: int):
    """Replace subtree

    Args:
        graph_1 (rostok_graph.GraphGrammar): 
        graph_2 (rostok_graph.GraphGrammar): 
        id_1 (int): 
        id_2 (int): 

    Returns:
        [GraphGrammar, GraphGrammar]: Changed graphs
    """
    subtree_1 = get_subtree_graph(graph_1, id_1)
    subtree_2 = get_subtree_graph(graph_2, id_2)

    graph_1 = remove_subtree_without_root(graph_1, id_1)
    graph_2 = remove_subtree_without_root(graph_2, id_2)

    graph_1 = add_subtree_using_rule(graph_1, id_1, subtree_2)
    graph_2 = add_subtree_using_rule(graph_2, id_2, subtree_1)
    return graph_1, graph_2


def subtree_crossover(graph_1: rostok_graph.GraphGrammar,
                      graph_2: rostok_graph.GraphGrammar,
                      max_depth: int = 0,
                      inplace: bool = False):
    if not inplace:
        graph_1 = deepcopy(graph_1)
        graph_2 = deepcopy(graph_2)

    id_1 = random.choice(list(graph_1.nodes()))
    available_ids_2 = available_node_ids_both_directions(graph_1, id_1, graph_2, callback_body)
    # Bypass
    if len(available_ids_2) == 0:
        return graph_1, graph_2

    id_2 = random.choice(available_ids_2)
    graph_1, graph_2 = subtree_crossover_select(graph_1, graph_2, id_1, id_2)
    return graph_1, graph_2