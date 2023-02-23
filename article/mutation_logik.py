from rostok.block_builder.blocks_utils import NodeFeatures
from rostok.graph_grammar.node import GraphGrammar, Node
from rule_sets.example_vocabulary import get_terminal_graph_three_finger, get_terminal_graph_no_joints
from copy import deepcopy
from rule_sets.example_vocabulary import node_vocab
import itertools
 

graph = get_terminal_graph_three_finger()
edges = list(graph.edges)
nodes_degree =  list(graph.out_degree)
sheet_list = [(i[0],) for i in nodes_degree if i[1] == 0]

edges.extend(sheet_list)

list_before = []
list_after =  []

def check_pass_node_types(functors: list, node):
    for feature in functors:
        if feature(node) is True:
            return True
    return False

def check_edge(functors_before: list, functors_after: list, edge: tuple[Node, Node]):
    in_passed = check_pass_node_types(functors_before, edges[0])
    out_passed = check_pass_node_types(functors_after, edges[0])
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



def available_for_del_body(graph : GraphGrammar) -> list[int]:
    dfs = graph.graph_partition_dfs()
    avalible_for_del_ids = []
    for branch in dfs:
        is_body = lambda x : NodeFeatures.is_body(graph.get_node_by_id(x))
        is_not_branching = lambda x : dict(graph.out_degree)[x] <= 1
        is_not_root = lambda x: graph.get_root_id() != x
        condition = lambda x: is_not_branching(x) and is_not_root(x)

        # Filter by Transform 
        splited_ids_by_transform = split_by_feature(branch, graph, NodeFeatures.is_transform)
        splited_ids_body_by_transform = [list(filter(is_body, list_ids)) for list_ids  in splited_ids_by_transform]
        avalible_for_del_on_length_transform = [list_ids for list_ids in splited_ids_body_by_transform if len(list_ids) >=2]
        flat_id_transform = list(*avalible_for_del_on_length_transform)
        
        # Filter by Joints
        splited_ids = split_by_feature(branch, graph, NodeFeatures.is_joint)
        splited_ids_body = [list(filter(is_body, list_ids)) for list_ids  in splited_ids]
        avalible_for_del_on_length = [list_ids for list_ids in splited_ids_body if len(list_ids) >=2]
        flat_ids = list(*avalible_for_del_on_length)
       
        # Filter by root and branching
        filtred_flat_ids = list(filter(condition, flat_ids))
        
        # Intersection filtred resaults 
        intersection_ids = list(set(filtred_flat_ids) & set(flat_id_transform))
        avalible_for_del_ids.extend(intersection_ids)
        
    return avalible_for_del_ids
    
