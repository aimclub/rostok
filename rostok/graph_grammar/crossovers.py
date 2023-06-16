from copy import deepcopy
import random
import rostok.graph_grammar.node as rostok_graph
from rostok.graph_grammar.node import Node
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import networkx as nx
from typing import Callable, List
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
    """Returns a list of ids for which the check_neighbors condition is true.
    check_neighbors iterate over all nodes and their neighbors
    Args:
        replacer_node (Node): _description_
        graph (rostok_graph.GraphGrammar): _description_
        check_neighbors (CALLBACK_NEIGHBORS_TYPE): _description_

    Returns:
        _type_: _description_
    """
    node_ids = list(graph.nodes)
    list_id_to_node = lambda x: [graph.get_node_by_id(i) for i in x]
    successors = [list_id_to_node(graph.successors(id_n)) for id_n in node_ids]
    predecessors = [list_id_to_node(graph.predecessors(id_n)) for id_n in node_ids]
    package = zip(node_ids, predecessors, successors)
    passed_ids = [
        node_ids_i for node_ids_i, predecessors_i, successors_i in package
        if check_neighbors(predecessors_i, successors_i, replacer_node)
    ]
    return passed_ids


def available_node_ids_both_directions(graph_1: rostok_graph.GraphGrammar, id_1: int,
                                       graph_2: rostok_graph.GraphGrammar,
                                       check_neighbors: CALLBACK_NEIGHBORS_TYPE):
    """Search for available replacement IDs. First, a check is performed for the node from graph_1 to graph_2. 
    Then, for each candidate, a substitution check is performed in graph_1

    Args:
        graph_1 (rostok_graph.GraphGrammar): 
        id_1 (int): 
        graph_2 (rostok_graph.GraphGrammar): 
        check_neighbors (CALLBACK_NEIGHBORS_TYPE): 

    Returns:
        _type_: _description_
    """
    replacer = graph_1.get_node_by_id(id_1)
    successors_1 = [graph_1.get_node_by_id(i) for i in graph_1.successors(id_1)]
    predecessors_1 = [graph_1.get_node_by_id(i) for i in graph_1.predecessors(id_1)]
    check_neighbors_replacer = partial(check_neighbors, predecessors_1, successors_1)
    available_ids_2 = available_node_ids_based_on_neighbors(replacer, graph_2, check_neighbors)

    available_ids_replacer_2 = [
        id_i for id_i in available_ids_2 if check_neighbors_replacer(graph_2.get_node_by_id(id_i))
    ]
    return available_ids_replacer_2


def check_neighbours(predecessors: List[Node], successors: List[Node], replacer: Node) -> bool:
    if NodeFeatures.is_body(replacer):
        return True
    elif NodeFeatures.is_transform(replacer):
        return check_neighbours_transform(predecessors)
    elif NodeFeatures.is_joint(replacer):
        return check_neighbours_joint(predecessors)
    return False


def check_neighbours_transform(predecessors: List[Node]) -> bool:
    #is_empty_list = lambda x : isinstance(x, Iterable) and len(x) == 0
    predecessors_avalible = [NodeFeatures.is_body, NodeFeatures.is_transform]
    if len(predecessors) == 0:
        return False
    predecessors_avalible_fun = lambda x: any([i(x) for i in predecessors_avalible])
    is_predecessors = all(map(predecessors_avalible_fun, predecessors))

    return is_predecessors


def check_neighbours_joint(predecessors: List[Node]) -> bool:
    if len(predecessors) == 0:
        return False
    predecessors_avalible = [NodeFeatures.is_body, NodeFeatures.is_transform]
    predecessors_avalible_fun = lambda x: any([i(x) for i in predecessors_avalible])
    is_predecessors = all(map(predecessors_avalible_fun, predecessors))

    return is_predecessors


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
    """ Swaps two subtrees selected by id

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


def subtree_crossover(
        graph_1: rostok_graph.GraphGrammar,
        graph_2: rostok_graph.GraphGrammar,
        max_depth: int = 0,  # pylint: disable=unused-argument 
        inplace: bool = False):
    """Swap random subtrees from graphs

    Args:
        graph_1 (rostok_graph.GraphGrammar): 
        graph_2 (rostok_graph.GraphGrammar): 
        max_depth (int, optional): Needed for compability with GOLEM. Defaults to 0.

    Returns:
        [GraphGrammar, GraphGrammar]: Changed graphs
    """
    if not inplace:
        graph_1 = deepcopy(graph_1)
        graph_2 = deepcopy(graph_2)

    id_1 = random.choice(list(graph_1.nodes()))
    available_ids_2 = available_node_ids_both_directions(graph_1, id_1, graph_2, check_neighbours)
    # Bypass
    if len(available_ids_2) == 0:
        return graph_1, graph_2

    id_2 = random.choice(available_ids_2)
    graph_1, graph_2 = subtree_crossover_select(graph_1, graph_2, id_1, id_2)
    return graph_1, graph_2
