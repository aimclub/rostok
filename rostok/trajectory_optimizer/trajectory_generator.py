import operator
from rostok.block_builder_api.block_parameters import FrameTransform
from rostok.graph_grammar.graph_utils import plot_graph_ids, plot_graph
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
    """
    Returns a list of unique root-based paths in the graph.

    Args:
        graph (GraphGrammar): The graph to extract root paths from.

    Returns:
        list[list[int]]: A list of unique root-based paths, where each path is 
        represented as a list of node IDs
    """
    root_paths = graph.get_sorted_root_based_paths()
    non_uniq = set()
    uniq_paths = []
    for path in root_paths:
        #[x for x in path if x not in non_uniq]
        new_path = list([x for x in path if x not in non_uniq])
        uniq_paths.append(new_path)
        non_uniq.update(path)
    return uniq_paths


def control_vector_linear(branch: list, start: float, multiplier: float):
    """
    Generates a linear control vector for a given branch.

    Args:
        branch (list): The branch to generate the control vector for.
        start (float): The initial control value.
        multiplier (float): The multiplier to be applied to the control value.

    Returns:
        list[float]: The linear control vector for the branch.
    """
    vec = []
    for num, _ in enumerate(branch):
        value = round(start + multiplier * num, 3)
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
    """Returns a generator that iterates over a pair branch 
    (based on get_sorted_root_based_paths) and coefficient.
    Iterate over branch with joints.
    Args:
        graph (GraphGrammar): _description_
        coefficients (list[tuple]): _description_

    Raises:
        Exception: _description_

    Yields:
        _type_: _description_
    """
    paths_id = uniq_root_paths(graph)
    is_joint_id = lambda id: NodeFeatures.is_joint(graph.get_node_by_id(id))
    branch_has_joint = lambda branch: any(map(is_joint_id, branch))
    paths_id_with_joints = list(filter(branch_has_joint, paths_id))
    if len(paths_id_with_joints) != len(coefficients):
        raise Exception("Coefficient vector must have the same size joint_root_paths result")
    start_multiplier_iter = coefficients.__iter__()
    get_nodes = lambda x: list(map(graph.get_node_by_id, x))
    paths_node = list(map(get_nodes, paths_id_with_joints))
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
    #unpucked = lambda x: control_vector_geom_prog(*x)
    unpucked = lambda x: control_vector_linear(*x)
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
    """Splits the node list into composite links, separating them at joint nodes.
    Filters out composite links that start with a joint node.
    Calculates the length of each composite link and returns a list of their lengths.

    Args:
        node_list (list[Node]): 

    Returns:
        list[float]: lengths
    """

    composite_links = list(split_before(node_list, NodeFeatures.is_joint))
    is_first_joint = lambda x: NodeFeatures.is_joint(x[0])
    composite_links_with_joints = list(filter(is_first_joint, composite_links))
    node_length = lambda x: block_length(x.block_blueprint)
    calculate_length = lambda x: sum(map(node_length, x[1:]))
    links_length = list(map(calculate_length, composite_links_with_joints))
    return links_length


def calculate_control_value_based_on_length(node_list: list[Node], start: float, multiplier: float):
    """Calls links_length_after_joint to get the lengths of composite links.
    Calculates control values based on the lengths of the composite link

    Args:
        node_list (list[Node]): bracnh(fingers)
        start (float): const value
        multiplier (float): multiplier for length

    Returns:
        list[float]: control values
    """

    links_length = links_length_after_joint(node_list)
    vec = [start + multiplier * l for l in links_length]
    return vec


def calculate_control_value_virtual_cable(node_list: list[Node], start: float, multiplier: float):
    """Calls links_length_after_joint to get the lengths of composite links.
    Calculates control values based on the lengths of the composite link.
    torque = start + length_value * multiplier 
    where length_value -- sum of all link lengths before joint and one link after

    Args:
        node_list (list[Node]): bracnh(fingers)
        start (float): const value
        multiplier (float): multiplier for length

    Returns:
        list[float]: control values
    """

    links_length = links_length_after_joint(node_list)
    links_length_accumulate = list(accumulate(links_length, operator.add))
    vec = [start + multiplier * l for l in links_length_accumulate]
    return vec

def tendon_like_control(graph: GraphGrammar, coefficients: list[tuple[float, float]]):
    """See the calculate_control_value_based_on_length function for understanding 
    how control actions are generated for a branch (finger)

    Args:
        graph (GraphGrammar): _description_
        coefficients (list[tuple[float, float]]): coefficients for linear function (a, b) a + bx 

    Returns:
        _type_: _description_
    """
    gen = path_node_iter(graph, coefficients)
    unpucked = lambda x: calculate_control_value_based_on_length(*x)
    joint_constants = list(map(unpucked, gen))
    res = {"initial_value": list(chain(*joint_constants))}
    return res

def cable_length_linear_control(graph: GraphGrammar, coefficients: list[tuple[float, float]]):
    """See the calculate_control_value_virtual_cable function for understanding 
    how control actions are generated for a branch (finger)

    Args:
        graph (GraphGrammar): _description_
        coefficients (list[tuple[float, float]]): coefficients for linear function (a, b) a + bx 

    Returns:
        _type_: _description_
    """
    gen = path_node_iter(graph, coefficients)
    unpucked = lambda x: calculate_control_value_virtual_cable(*x)
    joint_constants = list(map(unpucked, gen))
    res = {"initial_value": list(chain(*joint_constants))}
    return res