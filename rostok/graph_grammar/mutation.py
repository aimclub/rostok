from rostok.graph_grammar.node import GraphGrammar, Node
from copy import deepcopy
from random import random, choice, choices
from typing import Union
from copy import deepcopy
from itertools import chain

from rostok.graph_grammar.node_block_typing import NodeFeatures


def check_pass_node_types(functors: list, node):
    for feature in functors:
        if feature(node) is True:
            return True
    return False


def check_edge(functors_before: list, functors_after: list, edge: tuple[Node, Node]):
    in_passed = check_pass_node_types(functors_before, edge[0])
    out_passed = check_pass_node_types(functors_after, edge[1])
    return in_passed and out_passed


def split_by_feature(list_ids: list[int], graph: GraphGrammar, is_separator) -> list[list[int]]:
    res = []
    buf = []
    for element in list_ids:
        if not is_separator(graph.get_node_by_id(element)):
            buf.append(element)
        else:
            res.append(deepcopy(buf))
            buf = []

    res.append(deepcopy(buf))
    return res


def available_for_del_body(graph: GraphGrammar) -> list[int]:
    dfs = graph.graph_partition_dfs()
    avalible_for_del_ids = []
    for branch in dfs:
        is_body = lambda x: NodeFeatures.is_body(graph.get_node_by_id(x))
        is_not_branching = lambda x: dict(graph.out_degree)[x] <= 1
        is_not_root = lambda x: graph.get_root_id() != x
        condition = lambda x: is_not_branching(x) and is_not_root(x)

        # Filter by Transform
        splited_ids_by_transform = split_by_feature(branch, graph, NodeFeatures.is_transform)
        splited_ids_body_by_transform = [
            list(filter(is_body, list_ids)) for list_ids in splited_ids_by_transform
        ]
        avalible_for_del_on_length_transform = [
            list_ids for list_ids in splited_ids_body_by_transform if len(list_ids) >= 2
        ]
        flat_id_transform = list(chain.from_iterable(avalible_for_del_on_length_transform))

        # Filter by Joints
        splited_ids = split_by_feature(branch, graph, NodeFeatures.is_joint)
        splited_ids_body = [list(filter(is_body, list_ids)) for list_ids in splited_ids]
        avalible_for_del_on_length = [
            list_ids for list_ids in splited_ids_body if len(list_ids) >= 2
        ]
        flat_ids = list(chain.from_iterable(avalible_for_del_on_length))

        # Filter by root and branching
        filtred_flat_ids = list(filter(condition, flat_ids))

        # Intersection filtred resaults
        intersection_ids = list(set(filtred_flat_ids) & set(flat_id_transform))
        avalible_for_del_ids.extend(intersection_ids)

    return avalible_for_del_ids


def available_for_add_transform_edges(graph: GraphGrammar):
    edges = list(graph.edges)
    is_j0 = lambda x: not NodeFeatures.is_joint(graph.get_node_by_id(x[0]))
    return list(filter(is_j0, edges))


def available_for_add_joint_edges(graph: GraphGrammar):
    edges = list(graph.edges)
    is_j1 = lambda x: not NodeFeatures.is_joint(graph.get_node_by_id(x[1]))
    is_j0 = lambda x: not NodeFeatures.is_joint(graph.get_node_by_id(x[0]))
    is_t1 = lambda x: not NodeFeatures.is_transform(graph.get_node_by_id(x[1]))
    condition = lambda x: is_j1(x) and is_j0(x) and is_t1(x)
    return list(filter(condition, edges))


def add_node_between(edge: tuple[int, int], graph: GraphGrammar, node: Node):
    if not graph.has_edge(*edge):
        raise Exception(f"Graph not have edge: {edge}")
    id_aded_node = graph.get_uniq_id()
    graph.add_node(id_aded_node, Node=node)
    graph.remove_edge(*edge)
    graph.add_edge(id_aded_node, edge[1])
    graph.add_edge(edge[0], id_aded_node)


def add_node_after_leaf(psedo_edge: tuple[int], graph: GraphGrammar, node: Node):
    id_aded_node = graph.get_uniq_id()
    graph.add_node(id_aded_node, Node=node)
    graph.add_edge(psedo_edge[0], id_aded_node)


def del_node(node_id: int, graph: GraphGrammar):
    out_id_list = list(graph.neighbors(node_id))
    in_id_list = list(graph.predecessors(node_id))
    if len(in_id_list) == 0:
        raise Exception("Root node cannot be deleted")
    elif len(out_id_list) == 0:
        graph.remove_node(node_id)
    else:
        graph.remove_node(node_id)
        graph.add_edge(in_id_list[0], out_id_list[0])


def available_for_add_bodies_edges(graph) -> list[Union[tuple[int, int], tuple[
        int,
]]]:
    avalibale_edges = list(graph.edges)
    nodes_degree = list(graph.out_degree)
    sheet_list = [(i[0],) for i in nodes_degree if i[1] == 0]
    avalibale_edges.extend(sheet_list)
    return avalibale_edges


def get_random_node(nodes_list: list[Node],
                    type_distribution: tuple[float, float, float] = (1, 1, 1)):
    node_types = [NodeFeatures.is_body, NodeFeatures.is_joint, NodeFeatures.is_transform]
    curren_type = choices(node_types, type_distribution)[0]
    avalable_node = list(filter(curren_type, nodes_list))
    current_node = choice(avalable_node)
    return current_node


def add_node_mutation(node: Node, graph: GraphGrammar):
    if NodeFeatures.is_body(node):
        avalibale_edges = available_for_add_bodies_edges(graph)
        if len(avalibale_edges) == 0:
            return
        current_edge = choice(avalibale_edges)
        if len(current_edge) == 2:
            add_node_between(current_edge, graph, node)
        elif len(current_edge) == 1:
            add_node_after_leaf(current_edge, graph, node)
        else:
            raise Exception(f"Wrong edge: {current_edge}")
        return
    if NodeFeatures.is_transform(node):
        avalibale_edges = available_for_add_transform_edges(graph)
    elif NodeFeatures.is_joint(node):
        avalibale_edges = available_for_add_joint_edges(graph)
    else:
        raise Exception(f"Wrong node type: node")
    if len(avalibale_edges) == 0:
        return
    current_edge = choice(avalibale_edges)
    add_node_between(current_edge, graph, node)


def delete_node_mutation(node: Node, graph: GraphGrammar):
    is_body = lambda x: NodeFeatures.is_body(graph.get_node_by_id(x))
    is_joint = lambda x: NodeFeatures.is_joint(graph.get_node_by_id(x))
    is_transform = lambda x: NodeFeatures.is_transform(graph.get_node_by_id(x))

    if NodeFeatures.is_body(node):
        avalibale_node_id = available_for_del_body(graph)
    elif NodeFeatures.is_transform(node):
        avalibale_node_id = list(filter(is_transform, list(graph.nodes)))
    elif NodeFeatures.is_joint(node):
        avalibale_node_id = list(filter(is_joint, list(graph.nodes)))
    else:
        raise Exception(f"Wrong node type: node")

    if len(avalibale_node_id) == 0:
        return
    current_id = choice(avalibale_node_id)
    del_node(current_id, graph)


def add_mut(
    graph: GraphGrammar,
    nodes_list: list[Node],
    type_distribution: tuple[float, float, float] = (1, 1, 1)) -> GraphGrammar:
    """Mutation to add nodes while maintaining properties DAG and rostok builder.
    Added nodes do not spawn branches.

    Args:
        graph (GraphGrammar): Mutable graph, 
        nodes_list (list[Node]): Nodes for add
        type_distribution (tuple[float, float, float], optional): Relative weights (Body, Transform, Joint).
        Defaults to (1, 1, 1).

    Returns:
        GraphGrammar: Mutated graph
    """
    res_graph = deepcopy(graph)
    node = get_random_node(nodes_list, type_distribution)
    add_node_mutation(node, res_graph)
    return res_graph


def del_mut(
    graph: GraphGrammar,
    nodes_list: list[Node],
    type_distribution: tuple[float, float, float] = (1, 1, 1)) -> GraphGrammar:
    """Mutation to remove nodes while maintaining properties DAG and rostok builder

    Args:
        graph (GraphGrammar): Mutable graph
        nodes_list (list[Node]): Nodes for delete
        type_distribution (tuple[float, float, float], optional): Relative weights (Body, Transform, Joint). 
        Defaults to (1, 1, 1).

    Returns:
        GraphGrammar: Mutated graph
    """
    res_graph = deepcopy(graph)
    node = get_random_node(nodes_list, type_distribution)
    delete_node_mutation(node, res_graph)
    return res_graph
