import networkx as nx

from rostok.graph_grammar.node import GraphGrammar
from typing import Any, NamedTuple, Optional, TypedDict, Union
from rostok.graph_grammar.node_block_typing import NodeFeatures


def is_star_topology(graph: nx.DiGraph):
    degree = dict(graph.degree())
    root = [n for n, d in graph.in_degree() if d == 0][0]
    del degree[root]
    return all(value <= 2 for value in degree.values())

def get_leaf_body_ids(graph: GraphGrammar) -> list[int]:
    leaf_nodes = [
        node for node in graph.nodes() if graph.in_degree(node) != 0 and graph.out_degree(node) == 0
    ]
    return leaf_nodes

def nearest_joint(mech_graph: GraphGrammar, start_find_id: int, is_before: bool) -> Optional[int]:
    is_joint_id = lambda id: NodeFeatures.is_joint(mech_graph.get_node_by_id(id))
    branches = mech_graph.get_sorted_root_based_paths()
    cord = []
    for col, branch in enumerate(branches):
        if start_find_id in branch:
            row = branch.index(start_find_id)
            cord = [col, row]
            break
    if len(cord) == 0:
        raise Exception("Body id not find")
    target_finger = branches[cord[0]]

    if is_before:
        find_list = list(reversed(target_finger[:cord[1]]))
    else:
        find_list = target_finger[cord[1]:]

    for el in find_list:
        if is_joint_id(el):
            return el
    return None

def get_tip_ids(graph: GraphGrammar) -> list[int]:
    tip_bodies = []
    paths = graph.get_sorted_root_based_paths()
        for path in paths:
            tip = False
            path = path.reverse()
            for idx in path
                if NodeFeatures.is_body(self.graph.get_node_by_id(idx)):
                    tip = True
                    break
            if not tip:
                raise Exception('Attempt to find a tip on a path without bodies')
    return tip_bodies



