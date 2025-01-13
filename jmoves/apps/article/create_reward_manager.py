import multiprocessing
from networkx import Graph
import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator, visualize_constrains

from auto_robot_design.optimization.saver import (
    ProblemSaver, )
from auto_robot_design.description.builder import jps_graph2pinocchio_robot
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths, get_optimizing_joints
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_horizontal_trajectory, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward, PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ForceEllipsoidReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot, MIT_CHEETAH_PARAMS_DICT


def get_manager_preset_2_stair_climber(graph: Graph, optimizing_joints: dict, workspace_traj: np.ndarray, step_trajs: list[np.ndarray], squat_trajs: list[np.ndarray]):
    dict_trajectory_criteria = {
        "MASS": NeutralPoseMass()
    }
    # criteria calculated for each point on the trajectory
    dict_point_criteria = {
        "Effective_Inertia": EffectiveInertiaCompute(),
        "Actuated_Mass": ActuatedMass(),
        "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
    }

    crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
    error_calculator = PositioningErrorCalculator(
        error_key='error', jacobian_key="Manip_Jacobian")
    soft_constrain = PositioningConstrain(
        error_calculator=error_calculator, points=[workspace_traj])
    reward_manager = RewardManager(crag=crag)
    reward_manager.add_trajectory_aggregator
    acceleration_capability = MinAccelerationCapability(manipulability_key='Manip_Jacobian',
                                                        trajectory_key="traj_6d", error_key="error", actuated_mass_key="Actuated_Mass")

    heavy_lifting = HeavyLiftingReward(
        manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", error_key="error", mass_key="MASS")
    # reward_manager.agg_list =
    reward_manager.add_trajectory(step_trajs[0], 0)
    reward_manager.add_trajectory(step_trajs[1], 1)
    reward_manager.add_trajectory(step_trajs[2], 2)

    reward_manager.add_trajectory(squat_trajs[0], 10)
    reward_manager.add_trajectory(squat_trajs[1], 11)
    reward_manager.add_trajectory(squat_trajs[2], 12)

    reward_manager.add_reward(acceleration_capability, 0, weight=1)
    reward_manager.add_reward(acceleration_capability, 1, weight=1)
    reward_manager.add_reward(acceleration_capability, 2, weight=1)

    reward_manager.add_reward(heavy_lifting, 10, weight=1)
    reward_manager.add_reward(heavy_lifting, 11, weight=1)
    reward_manager.add_reward(heavy_lifting, 12, weight=1)

    reward_manager.add_trajectory_aggregator([0, 1, 2], 'mean')
    reward_manager.add_trajectory_aggregator([10, 11, 12], 'mean')

    reward_manager.close_trajectories()

    return reward_manager, crag, soft_constrain



def get_manager_preset_2_stair_single(graph: Graph, optimizing_joints: dict, workspace_traj: np.ndarray, step_trajs: list[np.ndarray], squat_trajs: list[np.ndarray]):
    dict_trajectory_criteria = {
        "MASS": NeutralPoseMass()
    }
    # criteria calculated for each point on the trajectory
    dict_point_criteria = {
        "Effective_Inertia": EffectiveInertiaCompute(),
        "Actuated_Mass": ActuatedMass(),
        "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
    }

    crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
    error_calculator = PositioningErrorCalculator(
        error_key='error', jacobian_key="Manip_Jacobian")
    soft_constrain = PositioningConstrain(
        error_calculator=error_calculator, points=[workspace_traj])
    reward_manager = RewardManager(crag=crag)
    reward_manager.add_trajectory_aggregator
    acceleration_capability = MinAccelerationCapability(manipulability_key='Manip_Jacobian',
                                                        trajectory_key="traj_6d", error_key="error", actuated_mass_key="Actuated_Mass")

    heavy_lifting = HeavyLiftingReward(
        manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", error_key="error", mass_key="MASS")
    # reward_manager.agg_list =
    reward_manager.add_trajectory(step_trajs[0], 0)
    reward_manager.add_trajectory(step_trajs[1], 1)
    reward_manager.add_trajectory(step_trajs[2], 2)

    reward_manager.add_trajectory(squat_trajs[0], 10)
    reward_manager.add_trajectory(squat_trajs[1], 11)
    reward_manager.add_trajectory(squat_trajs[2], 12)

    reward_manager.add_reward(acceleration_capability, 0, weight=1)
    reward_manager.add_reward(acceleration_capability, 1, weight=1)
    reward_manager.add_reward(acceleration_capability, 2, weight=1)

    reward_manager.add_reward(heavy_lifting, 10, weight=1)
    reward_manager.add_reward(heavy_lifting, 11, weight=1)
    reward_manager.add_reward(heavy_lifting, 12, weight=1)

    reward_manager.add_trajectory_aggregator([0, 1, 2], 'mean')
    reward_manager.add_trajectory_aggregator([10, 11, 12], 'mean')

    reward_manager.close_trajectories()

    return reward_manager, crag, soft_constrain
