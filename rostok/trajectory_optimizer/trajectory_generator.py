from collections.abc import Iterable

import numpy as np

from rostok.block_builder.blocks_utils import NodeFeatures
from rostok.graph_grammar.node import GraphGrammar, Node
from typing import Any
from functools import partial


def create_const_traj(torque_value, stop_time: float, time_step: float):
    timeseries_traj = []
    timeseries = list(np.arange(0, stop_time, time_step))
    traj = [torque_value for t in timeseries]
    timeseries_traj.append(timeseries)
    timeseries_traj.append(traj)
    return timeseries_traj


def create_step_traj(start: float,
                     stop_time: float,
                     time_step: float,
                     torque_value,
                     before_start_value: float = 0):
    timeseries_traj = []
    timeseries = list(np.arange(0, stop_time, time_step))
    traj = [torque_value if t > start else before_start_value for t in timeseries]
    timeseries_traj.append(timeseries)
    timeseries_traj.append(traj)
    return timeseries_traj


def create_dfs_joint(graph: GraphGrammar) -> list[list[Node]]:
    dfs_patrion_ids = graph.graph_partition_dfs()

    def get_node(node_id):
        return graph.get_node_by_id(node_id)

    dfs_patrion_node = [[get_node(node_id) for node_id in branch] for branch in dfs_patrion_ids]
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


def flat_to_dfs_joint(graph: GraphGrammar, flat: list[Any]) -> list[list[Any]]:
    if not isinstance(flat, Iterable):
        flat = [flat]
    flat_iter = iter(flat)

    dfs_list = []
    joint_dfs = create_dfs_joint(graph)
    for joint_dfs_row in joint_dfs:
        row = []
        for _ in joint_dfs_row:
            one_flat = next(flat_iter)
            row.append(one_flat)
        dfs_list.append(np.array(row))

    return dfs_list


def create_torque_traj_from_x(graph: GraphGrammar, x: list[float], stop_time: float,
                              time_step: float) -> list[list[Any]]:

    torque_traj = partial(create_const_traj, stop_time=stop_time, time_step=time_step)
    torque_trajs_flat = list(map(torque_traj, x))
    torque_trajs_dfs = flat_to_dfs_joint(graph, torque_trajs_flat)

    return torque_trajs_dfs

def create_step_torque_traj_from_x(graph: GraphGrammar, x: list[float], stop_time: float,
                              time_step: float, torque: float) -> list[list[Any]]:

    torque_traj = partial(create_step_traj, stop_time=stop_time, time_step=time_step, torque_value=torque)
    torque_trajs_flat = list(map(torque_traj, x))
    torque_trajs_dfs = flat_to_dfs_joint(graph, torque_trajs_flat)

    return torque_trajs_dfs


def create_control_from_graph(graph: GraphGrammar, torque_dict: dict[Node, float], stop_time: float,
                              time_step: float):
    dfs_j = create_dfs_joint(graph)
    dfs_traj_out = []
    for row in dfs_j:
        row_out = []
        for one_j in row:
            value = torque_dict[one_j]
            traj = create_const_traj(value, stop_time, time_step)
            row_out.append(traj)

        dfs_traj_out.append(np.array(row_out))
    return dfs_traj_out
