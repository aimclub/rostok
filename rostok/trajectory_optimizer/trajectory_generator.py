import numpy as np
from rostok.graph_grammar.node import GraphGrammar, Node
from rostok.block_builder.blocks_utils import NodeFeatures
from collections.abc import Iterable

def create_const_traj(torque_value, stop_time: float, time_step: float):
    timeseries_traj = []
    timeseries = list(np.arange(0, stop_time, time_step))
    traj = [torque_value for _ in timeseries]
    timeseries_traj.append(timeseries)
    timeseries_traj.append(traj)
    return timeseries_traj


def create_dfs_joint(graph: GraphGrammar) -> list[list[Node]]:
    dfs_patrion_ids = graph.graph_partition_dfs()
    def get_node(node_id): return graph.get_node_by_id(node_id)

    dfs_patrion_node = [[get_node(node_id) for node_id in branch]
                        for branch in dfs_patrion_ids]
    dfs_j = []
    number_trq = 0
    for branch in dfs_patrion_node:
        joint_branch = list(filter(NodeFeatures.is_joint, branch))
        # Strange things, for dect empty list
        # len([[]]) is 1
        len_joints = len(joint_branch)
        number_trq += len_joints
        if len_joints != 0:
            dfs_j.append(joint_branch)
    dfs_j.sort(key=len)
    return dfs_j


def create_torque_traj_from_x(graph: GraphGrammar, x: list[float], stop_time: float, time_step: float):
    if not isinstance(x, Iterable):
        x = [x]
    x_iter = iter(x)
        
    torque_traj = []
    joint_dfs = create_dfs_joint(graph)
    for branch in joint_dfs:
        control_one_branch = []
        for block in branch:
            one_torque = next(x_iter)
            control_one_branch.append(create_const_traj(
                one_torque, stop_time, time_step))
        torque_traj.append(np.array(control_one_branch))

    return torque_traj
