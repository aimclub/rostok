import os
from typing import Tuple, Union

import dill
import numpy as np
from pymoo.core.problem import ElementwiseProblem

from auto_robot_design.description.builder import jps_graph2pinocchio_robot, jps_graph2pinocchio_robot_3d_constraints
from auto_robot_design.generator.topologies.graph_manager_2l import GraphManager2L
from auto_robot_design.optimization.rewards.reward_base import (Reward, RewardManager)
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.description.mesh_builder.mesh_builder import MeshBuilder,  jps_graph2pinocchio_meshes_robot

def calculate_reward(graph, builder, crag, trajectory, reward, actuator, sf):
    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(graph, builder)
    constraint_error, trajectory_results = sf.calculate_constrain_error(crag, fixed_robot, free_robot)
    if constraint_error > 0:
        return constraint_error, []
    else:
        return reward.calculate(*trajectory_results, actuator = actuator)

