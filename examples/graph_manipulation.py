from example_vocabulary import (B_NODES, J_NODES, T_EXAMPLE,
                                get_terminal_graph_three_finger)

from rostok.graph_grammar.node import Node

graph_grab_torque_robot = get_terminal_graph_three_finger()

# Iterate over id and get Node object
node_list_plain = map(graph_grab_torque_robot.get_node_by_id,
                      graph_grab_torque_robot.get_ids_in_dfs_order())


# Needed because line of code is so long
def get_node(node_id):
    return graph_grab_torque_robot.get_node_by_id(node_id)


def is_joint(node: Node):
    return node in J_NODES


def is_body(node: Node):
    return node in B_NODES


def is_special_transform(node: Node):
    return node in T_EXAMPLE

# Return true if branch contain nodes from T_EXAMPLE
def branch_filter(branch: list[Node]):
    check_list = list(map(is_special_transform, branch))
    return any(check_list)


# List need to convert promise of iteration
j_node_list = list(filter(is_joint, node_list_plain))
print(node_list_plain)

dfs_patrion_ids = graph_grab_torque_robot.graph_partition_dfs()

# Nested list generator
dfs_patrion_node = [[get_node(node_id) for node_id in branch] for branch in dfs_patrion_ids]

# Iterate over dfs_patrion_node and form massive from suitable branch
branchs_with_special_t = list(filter(branch_filter, dfs_patrion_node))
