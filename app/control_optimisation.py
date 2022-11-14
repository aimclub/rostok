import context
import app_vocabulary
from scipy.optimize import differential_evolution
from engine.node import GraphGrammar
import pychrono as chrono
from numpy import arange
import numpy as np
from utils.blocks_utils import NodeFeatures
from utils.simulation_step import SimulationStepOptimization
from engine.node import Node
from typing import Union
import utils.criterion_calc as criterion

def create_dfs_joint(graph: GraphGrammar) -> Union[list[list[Node]], int]:
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

    return [dfs_j, number_trq]


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
 

def create_reward_fun(generated_graph: GraphGrammar, config_sys, stop_time, time_step_traj, flags, object_to_grab, list_B):
     
    
    init_pos = chrono.ChCoordsysD(object_to_grab.GetCoord())
    
    def reward(x):
        
        # Init object state
        object_to_grab.SetNoSpeedNoAcceleration()
        object_to_grab.SetCoord(init_pos)
       
        arr_traj = x
        sim = SimulationStepOptimization(
            arr_traj, generated_graph, object_to_grab)
        sim.set_flags_stop_simulation(flags)
        sim.change_config_system(config_sys)
        sim_output = sim.simulate_system(0.01)
        

        return -rew

    return reward