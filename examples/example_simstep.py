import numpy as np
import rostok.virtual_experiment.simulation_step as step
import rostok.virtual_experiment.robot as robot
import pychrono as chrono
import random

from rostok.criterion.flags_simualtions import FlagSlipout, FlagNotContact, FlagMaxTime
from example_ruleset import get_terminal_graph_ladoshaka, get_terminal_graph_two_finger, get_terminal_graph_three_finger
from rostok.block_builder.blocks_utils import NodeFeatures
from numpy import arange
from rostok.trajectory_optimizer.control_optimizer import num_joints
from rostok.trajectory_optimizer.trajectory_generator import create_dfs_joint

"""
    Example generate random constant torque
    Calculate all info about joint from graph
    
"""


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


mechs = [get_terminal_graph_three_finger,
         get_terminal_graph_ladoshaka, get_terminal_graph_two_finger]

for get_graph in mechs:
    G = get_graph()

    dfs_patrion_ids = G.graph_partition_dfs()
    def get_node(node_id): return G.get_node_by_id(node_id)

    dfs_patrion_node = [[get_node(node_id) for node_id in branch]
                        for branch in dfs_patrion_ids]
    dfs_j = create_dfs_joint(G)
    number_trq = num_joints(G)
    
    const_torque_koef = [random.random() for _ in range(number_trq)]
    arr_trj = create_torque_traj_from_x(dfs_j, const_torque_koef, 10, 0.1)

    chrono_system = chrono.ChSystemNSC()
    grab_robot = robot.Robot(G, chrono_system)

    joints = np.array(grab_robot.get_joints)

    config_sys = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}

    time_to_contact = 2
    time_without_contact = 0.2
    max_time = 2

    flags = [FlagMaxTime(max_time)]

    times_step = 1e-3

    grab_obj_mat = chrono.ChMaterialSurfaceNSC()
    grab_obj_mat.SetFriction(0.5)
    grab_obj_mat.SetDampingF(0.1)
    obj = chrono.ChBodyEasyBox(0.2, 0.2, 0.6, 1000, True, True, grab_obj_mat)
    obj.SetCollide(True)
    obj.SetPos(chrono.ChVectorD(0, 1.2, 0))

    sim = step.SimulationStepOptimization(arr_trj, G, obj)
    sim.set_flags_stop_simulation(flags)
    sim.change_config_system(config_sys)
    sim_output = sim.simulate_system(times_step, True)
