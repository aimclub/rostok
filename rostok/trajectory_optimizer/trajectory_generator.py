from rostok.graph_grammar.graph_utils import plot_graph_ids
from rostok.graph_grammar.node import GraphGrammar
import networkx as nx
from collections import Counter
from itertools import chain

from rostok.graph_grammar.node_block_typing import NodeFeatures, get_joint_vector_from_graph


def uniq_root_paths(graph: GraphGrammar) -> list[list[int]]:
    root_paths = graph.get_sorted_root_based_paths()
    non_uniq = set()
    uniq_paths = []
    for path in root_paths:
        new_path = list(set(path) - non_uniq)
        uniq_paths.append(new_path)
        non_uniq.update(path)
    return uniq_paths


def control_vector_linear(branch: list, start: float, multiplier: float):
    vec = []
    for num, _ in enumerate(branch):
        value = start + multiplier * num
        vec.append(value)
    return vec


def control_vector_geom_prog(branch: list, start: float, multiplier: float):
    vec = []
    for num, _ in enumerate(branch):
        value = start * multiplier**num
        vec.append(value)
    return vec


def joint_root_paths(graph: GraphGrammar):
    branches = uniq_root_paths(graph)
    is_joint_id = lambda id: NodeFeatures.is_joint(graph.get_node_by_id(id))
    joint_branchs = [[ids for ids in path if is_joint_id(ids)] for path in branches]
    joint_branchs = list(filter(bool, joint_branchs))
    return joint_branchs


def branch_control_generator(graph: GraphGrammar, coefficients: list[tuple]):
    """Returns a generator that iterates over a pair branch 
    (based on get_sorted_root_based_paths) and coefficient.

    Args:
        graph (GraphGrammar): 
        koeficients (list[tuple]): koeficients for one branch

    Raises:
        Exception: Coefficient vector must have the same size joint_root_paths result

    Yields:
        _type_: _description_
    """
    joint_branchs = joint_root_paths(graph)
    if len(joint_branchs) != len(coefficients):
        raise Exception("Coefficient vector must have the same size joint_root_paths result")
    start_multiplier_iter = coefficients.__iter__()
    for branch in joint_branchs:
        start_multiplier = next(start_multiplier_iter)
        yield branch, *start_multiplier


def linear_control(graph: GraphGrammar, coefficients: list[tuple[float, float]]):
    """_summary_

    Args:
        graph (GraphGrammar): _description_
        coefficients (list[tuple[float, float]]): _description_

    Returns:
        _type_: _description_
    """
    gen = branch_control_generator(graph, coefficients)
    unpucked = lambda x: control_vector_geom_prog(*x)
    joint_constants = list(map(unpucked, gen))
    res = {"initial_value" : list(chain(*joint_constants))}
    return res


