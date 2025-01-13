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
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths, CalculateMultiCriteriaProblem, get_optimizing_joints
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.analyze_squat_history import get_sample_torque_traj_from_sample_multi
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_horizontal_trajectory, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward, PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ForceEllipsoidReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot, MIT_CHEETAH_PARAMS_DICT
from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
import numpy as np

import matplotlib.pyplot as plt

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop
from auto_robot_design.optimization.analyze import get_optimizer_and_problem, get_pareto_sample_linspace, get_pareto_sample_histogram, get_urdf_from_problem

paths = [
        "results\\multi_opti_preset2\\topology_0_2024-05-29_18-48-58",
         "results\\multi_opti_preset2\\topology_1_2024-05-29_19-37-36",
         "results\\multi_opti_preset2\\topology_3_2024-05-29_23-01-44",
         "results\\multi_opti_preset2\\topology_4_2024-05-29_23-46-17",
         "results\\multi_opti_preset2\\topology_5_2024-05-30_00-32-21",
         "results\\multi_opti_preset2\\topology_7_2024-05-30_01-15-44",
         "results\\multi_opti_preset2\\topology_8_2024-05-30_10-40-12",
         ]

# for path_i in paths:
#     get_sample_torque_traj_from_sample_multi(path_i, False)
get_sample_torque_traj_from_sample_multi("results\\multi_opti_preset2222\\topology_8_2024-05-30_10-40-12", True)