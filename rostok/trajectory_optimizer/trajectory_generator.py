from rostok.block_builder_api.block_parameters import FrameTransform
from rostok.graph_grammar.graph_utils import plot_graph_ids
from rostok.graph_grammar.node import GraphGrammar, Node
import networkx as nx
import numpy as np
from collections import Counter
from itertools import accumulate, chain
from rostok.graph_grammar.node_block_typing import NodeFeatures, get_joint_vector_from_graph
from rostok.block_builder_api import block_blueprints, easy_body_shapes
from rostok.library.rule_sets.simple_designs import get_two_link_three_finger
from more_itertools import split_before

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


def path_control_generator(graph: GraphGrammar, coefficients: list[tuple]):
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

def path_node_iter(graph: GraphGrammar, coefficients: list[tuple]):
    paths_id = uniq_root_paths(graph)
    if len(paths_id) != len(coefficients):
        raise Exception("Coefficient vector must have the same size joint_root_paths result")
    start_multiplier_iter = coefficients.__iter__()
    get_nodes = lambda x : map(graph.get_node_by_id, x)
    paths_node = list(map(get_nodes, paths_id))
    for branch in paths_node:
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
    gen = path_control_generator(graph, coefficients)
    unpucked = lambda x: control_vector_geom_prog(*x)
    joint_constants = list(map(unpucked, gen))
    res = {"initial_value": list(chain(*joint_constants))}
    return res


def block_length(blueprint: block_blueprints.ALL_BLUEPRINT_TYPE) -> float:
    if isinstance(blueprint, block_blueprints.PrimitiveBodyBlueprint):
        if isinstance(blueprint.shape, easy_body_shapes.Box):
            return blueprint.shape.length_y
        else:
            raise Exception("Block length not implemented for other shapes")
    elif isinstance(blueprint, block_blueprints.TransformBlueprint):
        return np.linalg.norm(blueprint.transform.position)

    else:
        raise Exception("Block length not implemented for other blueprints")


def calculate_length_and_filter_joint(node_list: list[Node]):
    calculate_length = lambda x: x if isinstance(x.block_blueprint, block_blueprints.
                                                 RevolveJointBlueprint) else block_length(
                                                     x.block_blueprint)
    length_and_joint = list(map(calculate_length, node_list))
    return length_and_joint



def links_length_after_joint(node_list: list[Node]) -> list[float]:
    composite_links = list(split_before(node_list,  NodeFeatures.is_joint))
    is_first_joint = lambda x: NodeFeatures.is_joint(x[0])
    composite_links_with_joints = list(filter(is_first_joint, composite_links))
    node_length = lambda x: block_length(x.block_blueprint)
    calculate_length = lambda x: sum(map(node_length, x[1:]))
    links_length = list(map(calculate_length, composite_links_with_joints))
    return links_length

def path_coef_tendon_like(node_list: list[Node], start: float, multiplier: float):
    links_length = links_length_after_joint(node_list)
    vec = []
    for l in range(len(links_length)):
        value = start + multiplier * l
        vec.append(value)
    return vec

def tendon_like_control(graph: GraphGrammar, coefficients: list[tuple[float, float]]):
    gen = path_node_iter(graph, coefficients)
    unpucked = lambda x: path_coef_tendon_like(*x)
    joint_constants = list(map(unpucked, gen))
    res = {"initial_value": list(chain(*joint_constants))}
    return res


    
