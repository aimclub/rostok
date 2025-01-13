from auto_robot_design.pinokla.criterion_agregator import load_criterion_traj
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
from pathlib import Path


def get_all_files_in_dir(directory):
    try:
        # Create a Path object for the directory
        path = Path(directory)

        # Use the glob method to match all files in the directory
        files = [str(file) for file in path.glob('*') if file.is_file()]

        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def get_metrics(loaded_dict: dict):
    trq_1 = loaded_dict["tau"][:, 0]
    trq_2 = loaded_dict["tau"][:, 1]
    max_1 = np.max(np.abs(trq_1))
    max_2 = np.max(np.abs(trq_2))

    agr_max = max([max_1, max_2])

    mean_1 = np.mean(np.abs(trq_1))
    mean_2 = np.mean(np.abs(trq_2))

    agr_mean = mean_1 + mean_2

    max_diff_1 = np.max(np.abs(np.diff(trq_1)))
    max_diff_2 = np.max(np.abs(np.diff(trq_2)))

    agr_diff = max([max_diff_1, max_diff_2])
    return agr_mean, agr_max, agr_diff, loaded_dict["Reward"], loaded_dict["X"]


def get_all_vector_metrics(directory):
    sim_res_files = get_all_files_in_dir(directory)
    sim_res = list(map(load_criterion_traj, sim_res_files))
    agr_mean_list = []
    agr_max_list = []
    agr_diff_list = []
    reword_list = []
    param_x_list = []
    for sim_res_i in sim_res:
        agr_mean, agr_max, agr_diff, reword, param_x = get_metrics(sim_res_i)
        agr_mean_list.append(agr_mean)
        agr_max_list.append(agr_max)
        agr_diff_list.append(agr_diff)
        reword_list.append(reword)
        param_x_list.append(param_x)
    return agr_mean_list, agr_max_list, agr_diff_list, reword_list, param_x_list


PATH_CS = "results\\multi_opti_preset2\\topology_8_2024-05-30_10-40-12\\squat_compare"
agr_mean_list, agr_max_list, agr_diff_list, reword_list, param_x_list = get_all_vector_metrics(
    PATH_CS)
save_p = Path(PATH_CS + "/" + "plots" + "/")
save_p.mkdir(parents=True, exist_ok=True)

plt.figure()
plt.scatter(np.array(reword_list)[:, 0], np.array(
    reword_list)[:, 1], c=agr_mean_list, cmap="rainbow")
plt.colorbar()
plt.title("Mean torque in squat_sim on Pareto front")
plt.xlabel("ACC Capability")
plt.ylabel("HeavyLifting")

save_current1 = save_p / "Mean_torque_in_squat_sim_on_Pareto_front.svg"
save_current2 = save_p / "Mean_torque_in_squat_sim_on_Pareto_front.png"
plt.savefig(save_current1)
plt.savefig(save_current2)

plt.figure()
plt.scatter(np.array(reword_list)[:, 0], np.array(
    reword_list)[:, 1], c=agr_max_list, cmap="rainbow")
plt.colorbar()
plt.title("Max torque in squat_sim on Pareto front")
plt.xlabel("ACC Capability")
plt.ylabel("HeavyLifting")

save_current1 = save_p / "Max_torque_in_squat_sim_on_Pareto_front.svg"
save_current2 = save_p / "Max_torque_in_squat_sim_on_Pareto_front.png"
plt.savefig(save_current1)
plt.savefig(save_current2)


plt.figure()
plt.scatter(np.array(reword_list)[:, 0], np.array(
    reword_list)[:, 1], c=agr_diff_list, cmap="rainbow")
plt.colorbar()
plt.title("Max torque diff in squat_sim on Pareto front")
plt.xlabel("ACC Capability")
plt.ylabel("HeavyLifting")
plt.savefig("Max torque diff in squat_sim on Pareto front")
save_current1 = save_p / "Max_torque_diff_in_squat_sim_on_Pareto_front.svg"
save_current2 = save_p / "Max_torque_diff_in_squat_sim_on_Pareto_front.png"
plt.savefig(save_current1)
plt.savefig(save_current2)
plt.show()
