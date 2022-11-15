import context
import app_vocabulary
from scipy.optimize import differential_evolution
from engine.node import GraphGrammar
import pychrono as chrono
from numpy import arange
import numpy as np
from utils.blocks_utils import NodeFeatures
from utils.simulation_step import SimulationStepOptimization,  SimulationDataBlock
from engine.node import Node
from typing import Union
import utils.criterion_calc as criterion
from utils.flags_simualtions import FlagSlipout, FlagNotContact, FlagMaxTime
from scipy.optimize import shgo
import random


def create_multidimensional_bounds(graph: GraphGrammar, one_d_bound: tuple[float, float]):
    num = num_joints(graph)
    multidimensional_bounds = []
    for i in range(num):
        multidimensional_bounds.append(one_d_bound)

    return multidimensional_bounds


def num_joints(graph: GraphGrammar):
    line_order = graph.get_ids_in_dfs_order()
    list_nodes = list(map(graph.get_node_by_id, line_order))
    return sum(map(NodeFeatures.is_joint, list_nodes))


def get_object_to_grasp():
    grab_obj_mat = chrono.ChMaterialSurfaceNSC()
    grab_obj_mat.SetFriction(0.5)
    grab_obj_mat.SetDampingF(0.1)
    obj = chrono.ChBodyEasyBox(0.2, 0.2, 0.6, 1000, True, True, grab_obj_mat)
    obj.SetCollide(True)
    obj.SetPos(chrono.ChVectorD(0, 1.2, 0))
    return obj


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

    return dfs_j


def create_const_traj(torque_value, stop_time: float, time_step: float):
    timeseries_traj = []
    timeseries = list(arange(0, stop_time, time_step))
    traj = [torque_value for _ in timeseries]
    timeseries_traj.append(timeseries)
    timeseries_traj.append(traj)
    return timeseries_traj


def create_torque_traj_from_x(joint_dfs, x: list[float], stop_time: float, time_step: float):
    x_iter = iter(x)
    torque_traj = []
    for branch in joint_dfs:
        control_one_branch = []
        for block in branch:
            one_torque = next(x_iter)
            control_one_branch.append(create_const_traj(
                one_torque, stop_time, time_step))
        torque_traj.append(np.array(control_one_branch))

    return torque_traj


def grab_crtitrion(sim_output: dict[int, SimulationDataBlock], grab_robot, node_feature: list[list[Node]]):
    gait = 2.5
    weight = [1, 1, 1, 1]
    j_nodes = criterion.nodes_division(
        grab_robot, node_feature[1])
    b_nodes = criterion.nodes_division(
        grab_robot, node_feature[0])
    rb_nodes = criterion.sort_left_right(
        grab_robot, node_feature[3], node_feature[0])
    lb_nodes = criterion.sort_left_right(
        grab_robot, node_feature[2], node_feature[0])

    return criterion.criterion_calc(sim_output, b_nodes, j_nodes, rb_nodes, lb_nodes, weight, gait)


def grab_crtitrion_with_context(sim_output: dict[int, SimulationDataBlock], grab_robot):
    return grab_crtitrion(sim_output, grab_robot, app_vocabulary.node_features)


def create_reward_fun(generated_graph: GraphGrammar, config_sys, stop_time, time_step_traj, flags, object_to_grab, criterion_callback):

    init_pos = chrono.ChCoordsysD(object_to_grab.GetCoord())
    dfs_j = create_dfs_joint(generated_graph)

    def reward(x):
        # Init object state
        is_vis = random.random() > 0.95
        object_to_grab.SetNoSpeedNoAcceleration()
        object_to_grab.SetCoord(init_pos)
        arr_traj = create_torque_traj_from_x(
            dfs_j, x, stop_time, time_step_traj)
        sim = SimulationStepOptimization(
            arr_traj, generated_graph, object_to_grab)
        sim.set_flags_stop_simulation(flags)
        sim.change_config_system(config_sys)
        sim_output = sim.simulate_system(0.001, False)
        rew = criterion_callback(sim_output, sim.grab_robot)
        print(rew)
        return rew

    return reward


mechanism_graph = app_vocabulary.get_three_finger_graph()


flags = [FlagMaxTime(2)]

times_step = 1e-3
bound = (-5, 5)

config_sys = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
obj = get_object_to_grasp()

rewa = create_reward_fun(mechanism_graph, config_sys,
                         2, 0.01, flags, obj, grab_crtitrion_with_context)
multi_bound = create_multidimensional_bounds(mechanism_graph, bound)
result = shgo(rewa, multi_bound, n=10, iters=2)

print(result.x)
print(result.fun)
