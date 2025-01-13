import multiprocessing
import time
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
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward, PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ForceEllipsoidReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot, MIT_CHEETAH_PARAMS_DICT
from apps.article import create_reward_manager
from apps.article import traj_graph_setup
from pymoo.algorithms.moo.age2 import AGEMOEA2


def run_one_optic_topology(topology_num):

    POP_SIZE = 32
    GEN_SIZE = 90
    N_PROCESS = 4

    graph, optimizing_joints, constrain_dict, builder, step_trajs, squat_trajs, workspace_trajectory = traj_graph_setup.get_graph_and_traj(
        topology_num)
    reward_manager, crag, soft_constrain = create_reward_manager.get_manager_preset_2_stair_climber(
        graph, optimizing_joints, workspace_traj=workspace_trajectory, step_trajs=step_trajs, squat_trajs=squat_trajs)

    pool = multiprocessing.Pool(N_PROCESS)
    runner = StarmapParallelization(pool.starmap)

    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    problem = CalculateMultiCriteriaProblem(graph, builder=builder,
                                            jp2limits=optimizing_joints,
                                            crag=crag,
                                            soft_constrain=soft_constrain,
                                            rewards_and_trajectories=reward_manager,
                                            runner=runner,
                                            Actuator=actuator)

    saver = ProblemSaver(
        problem, "multi_opti_preset2\\topology_"+str(topology_num), True)
    saver.save_nonmutable()

    algorithm = AGEMOEA2(pop_size=POP_SIZE, save_history=True)
    optimizer = PymooOptimizer(problem, algorithm, saver)
    start = time.time()
    res = optimizer.run(
        True, **{
            "seed": 5,
            "termination": ("n_gen", GEN_SIZE),
            "verbose": True
        })
    elap = (time.time() - start) / 60
    print(f"Proshlo: {elap} minutes")


if __name__ == '__main__':
    run_one_optic_topology(8)
        # except:
        #     print(f"Fall optimization topology {i}")