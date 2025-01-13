from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable
from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
import numpy as np

import matplotlib.pyplot as plt

from auto_robot_design.description.actuators import MyActuator_RMD_MT_RH_17_100_N, RevoluteActuator, t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths, CalculateMultiCriteriaProblem
from auto_robot_design.optimization.saver import load_checkpoint
from auto_robot_design.pinokla.criterion_agregator import load_criterion_traj, save_criterion_traj
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop
import dill
import os
from auto_robot_design.description.utils import draw_joint_point

def pareto_front_with_indices(data):
    """
    Extract the Pareto front and their indices from a set of multi-objective data points.

    Parameters:
    data (np.ndarray): A 2D array where each row is a solution and each column is an objective.

    Returns:
    tuple: A tuple containing:
        - np.ndarray: A 2D array containing the Pareto front.
        - np.ndarray: A 1D array containing the indices of the Pareto front solutions.
    """
    # Initialize a boolean array to mark dominated solutions
    is_dominated = np.zeros(data.shape[0], dtype=bool)

    # Compare each solution with every other solution
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                if all(data[j] <= data[i]) and any(data[j] < data[i]):
                    is_dominated[i] = True
                    break

    # Extract non-dominated solutions and their indices
    pareto_front = data[~is_dominated]
    pareto_indices = np.where(~is_dominated)[0]

    return pareto_front, pareto_indices


def reward_vel_with_context(sim_hopp: SimulateSquatHop, robo_urdf: str,
                            joint_description: dict, loop_description: dict,
                            x: float):
    NUMBER_LAST_VALUE = 150
    TRQ_DEVIDER = 1000

    q_act, vq_act, acc_act, tau = sim_hopp.simulate(robo_urdf,
                                                    joint_description,
                                                    loop_description, control_coefficient=float(x))
    trj_f = sim_hopp.create_traj_equation()
    t = np.linspace(0, sim_hopp.squat_hop_parameters.total_time, len(q_act))
    q_vq_acc_des = np.array(list(map(trj_f, t)))
    vq_des = q_vq_acc_des[:, 1]
    tail_vq_des = vq_des[-NUMBER_LAST_VALUE:]
    tail_vq_act = vq_act[-NUMBER_LAST_VALUE:]
    tail_vq_act = tail_vq_act[:, 0]
    vq_erorr = np.mean(np.abs(tail_vq_des - tail_vq_act))
    return vq_erorr


def reward_vel_with_context(sim_hopp: SimulateSquatHop, robo_urdf: str,
                            joint_description: dict, loop_description: dict,
                            x: float):
    NUMBER_LAST_VALUE = 100
    TRQ_DEVIDER = 1000

    q_act, vq_act, acc_act, tau = sim_hopp.simulate(robo_urdf,
                                                    joint_description,
                                                    loop_description, control_coefficient=float(x))
    trj_f = sim_hopp.create_traj_equation()
    t = np.linspace(0, sim_hopp.squat_hop_parameters.total_time, len(q_act))
    q_vq_acc_des = np.array(list(map(trj_f, t)))
    vq_des = q_vq_acc_des[:, 1]
    tail_vq_des = vq_des[-NUMBER_LAST_VALUE:]
    tail_vq_act = vq_act[-NUMBER_LAST_VALUE:]
    tail_vq_act = tail_vq_act[:, 0]
    vq_erorr = np.mean(np.abs(tail_vq_des - tail_vq_act))
    # print(f"Errror:{vq_erorr}, x: {x}")

    return vq_erorr


def min_vel_error_control_brute_force(min_fun: Callable[[float], float]):
    x_vec = np.linspace(0.65, 0.9, 10)
    errors = []
    for x in x_vec:
        try:
            res = min_fun(x)
        except:
            res = 1
        errors.append(res)
    x_and_err = zip(x_vec, errors)
    def key_fun(tup): return tup[1]
    min_x_and_error = min(x_and_err, key=key_fun)
    return min_x_and_error


def min_pos_error_control_brute_force(min_fun: Callable[[float], float]):
    x_vec = np.linspace(0.65, 0.9, 10)
    errors = []
    for x in x_vec:
        try:
            res = min_fun(x)
        except:
            res = 1
        errors.append(res)
    x_and_err = zip(x_vec, errors)
    def key_fun(tup): return tup[1]
    min_x_and_error = min(x_and_err, key=key_fun)
    return min_x_and_error


def get_history_and_problem(path):
    problem = CalculateMultiCriteriaProblem.load(
        path)
    checklpoint = load_checkpoint(path)

    optimizer = PymooOptimizer(problem, checklpoint)
    optimizer.load_history(path)
    return optimizer.history, problem


def get_optimizer_and_problem(path) -> tuple[PymooOptimizer, CalculateMultiCriteriaProblem]:
    problem = CalculateMultiCriteriaProblem.load(
        path)
    checklpoint = load_checkpoint(path)

    optimizer = PymooOptimizer(problem, checklpoint)
    optimizer.load_history(path)
    res = optimizer.run()

    return optimizer, problem, res


def get_pareto_sample_linspace(res, sample_len: int):

    sample_indices = np.linspace(0, len(res.F) - 1, sample_len, dtype=int)
    sample_x = res.X[sample_indices]
    sample_F = res.F[sample_indices]

    return sample_x, sample_F


def get_pareto_sample_histogram(res, sample_len: int):
    """Histogram uses 0 from reword vector

    Args:
        res (_type_): _description_
        sample_len (int): _description_

    Returns:
        _type_: _description_
    """
    rewards = res.F
    _, bins_edg = np.histogram(rewards[:, 0], sample_len)
    bin_indices = np.digitize(rewards[:, 0], bins_edg, right=True)
    bins_set_id = [np.where(bin_indices == i)[0]
                   for i in range(1, len(bins_edg))]
    best_in_bins = [i[0] for i in bins_set_id if len(i) > 0]
    sample_F = rewards[best_in_bins]
    sample_X = res.X[best_in_bins]
    return sample_X, sample_F


def get_urdf_from_problem(sample_X: np.ndarray, problem: CalculateMultiCriteriaProblem):
    problem.mutate_JP_by_xopt(problem.initial_xopt)
    graphs = []
    urdf_j_des_l_des = []
    for x_i in sample_X:
        problem.mutate_JP_by_xopt(x_i)
        mutated_graph = deepcopy(problem.graph)

        robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
            mutated_graph, problem.builder)
        graphs.append(mutated_graph)
        urdf_j_des_l_des.append(
            (robo_urdf, joint_description, loop_description))
    return graphs, urdf_j_des_l_des


def get_sorted_history(history: dict):
    rewards = np.array(history["F"]).flatten()
    x_value = np.array(history["X"])
    ids_sorted = np.argsort(rewards)
    sorted_reward = rewards[ids_sorted]
    sorted_x_values = x_value[ids_sorted]
    return sorted_reward, sorted_x_values


def get_histogram_data(rewards):
    NUMBER_BINS = 10
    _, bins_edg = np.histogram(rewards, NUMBER_BINS)
    bin_indices = np.digitize(rewards, bins_edg, right=True)
    bins_set_id = [np.where(bin_indices == i)[0]
                   for i in range(1, len(bins_edg))]
    return bins_set_id


def get_tested_reward_and_x(sorted_reward: np.ndarray, sorted_x_values: np.ndarray):
    bins_set_id = get_histogram_data(sorted_reward)
    best_in_bins = [i[0] for i in bins_set_id]
    return sorted_reward[best_in_bins], sorted_x_values[best_in_bins]


def get_sample_torque_traj_from_sample(path):

    PATH_TO_LOAD_OPTIMISATION_RES = path
    PATH_CURRENT_MECH = Path(PATH_TO_LOAD_OPTIMISATION_RES) / "squat_compare"
    history, problem = get_history_and_problem(PATH_TO_LOAD_OPTIMISATION_RES)
    sorted_reward, sorted_x_values = get_sorted_history(history)
    rewards_sample, x_sample = get_tested_reward_and_x(
        sorted_reward, sorted_x_values)

    graphs = []
    urdf_j_des_l_des = []
    for x_i in x_sample:
        problem.mutate_JP_by_xopt(x_i)
        mutated_graph = deepcopy(problem.graph)

        robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
            mutated_graph, problem.builder)
        graphs.append(mutated_graph)
        urdf_j_des_l_des.append(
            (robo_urdf, joint_description, loop_description))

    sqh_p = SquatHopParameters(hop_flight_hight=0.2,
                               squatting_up_hight=0.0,
                               squatting_down_hight=-0.38,
                               total_time=0.8)
    hoppa = SimulateSquatHop(sqh_p)
    for i, (robo_urdf_i, joint_description_i, loop_description_i) in enumerate(urdf_j_des_l_des):
        opti = partial(reward_vel_with_context, hoppa, robo_urdf_i, joint_description_i,
                       loop_description_i)
        res = min_vel_error_control_brute_force(opti)
        q_act, vq_act, acc_act, tau = hoppa.simulate(robo_urdf_i,
                                                     joint_description_i,
                                                     loop_description_i,
                                                     control_coefficient=res[0],
                                                     is_vis=False)
        max1t = max(np.abs(tau[:, 0]))
        max2t = max(np.abs(tau[:, 1]))
        trj_f = hoppa.create_traj_equation()
        t = np.linspace(0, sqh_p.total_time, len(q_act))
        list__234 = np.array(list(map(trj_f, t)))

        saved_dict = {}
        saved_dict["ControlConst"] = res[0]
        saved_dict["ControlError"] = res[1]
        saved_dict["Reward"] = rewards_sample[i]
        saved_dict["X"] = x_sample[i]
        saved_dict["pos_act"] = q_act[:, 0]
        saved_dict["v_act"] = vq_act[:, 0]
        saved_dict["acc_act"] = acc_act[:, 0]
        saved_dict["tau"] = tau
        saved_dict["HopParams"] = sqh_p
        print(
            f"Max 1 act: {max1t}, Max 2 act: {max2t}, Reward:{rewards_sample[i]}, Error vel: {res[1]}")
        save_criterion_traj(robo_urdf_i, PATH_CURRENT_MECH,
                            loop_description_i, joint_description_i, saved_dict)


def get_sample_torque_traj_from_sample_multi(path, is_vis = False):

    path_to_save_result = Path(path) / "squat_compare"
    optimizer, problem, res = get_optimizer_and_problem(path)
    sample_X, sample_F = get_pareto_sample_histogram(res, 10)
    graphs, urdf_j_des_l_des = get_urdf_from_problem(sample_X, problem)
    sqh_p = SquatHopParameters(hop_flight_hight=0.10,
                               squatting_up_hight=0.0,
                               squatting_down_hight=-0.04,
                               total_time=0.2)
    hoppa = SimulateSquatHop(sqh_p)


    for i, (robo_urdf_i, joint_description_i, loop_description_i) in enumerate(urdf_j_des_l_des):
        opti = partial(reward_vel_with_context, hoppa, robo_urdf_i, joint_description_i,
                       loop_description_i)
        res = min_vel_error_control_brute_force(opti)
        q_act, vq_act, acc_act, tau = hoppa.simulate(robo_urdf_i,
                                                     joint_description_i,
                                                     loop_description_i,
                                                     control_coefficient=res[0],
                                                     is_vis=is_vis)
        max1t = max(np.abs(tau[:, 0]))
        max2t = max(np.abs(tau[:, 1]))
        trj_f = hoppa.create_traj_equation()
        t = np.linspace(0, sqh_p.total_time, len(q_act))
        list__234 = np.array(list(map(trj_f, t)))

        saved_dict = {}
        saved_dict["Graph"] = graphs[i]
        saved_dict["ControlConst"] = res[0]
        saved_dict["ControlError"] = res[1]
        saved_dict["Reward"] = sample_F[i]
        saved_dict["X"] = sample_X[i]
        saved_dict["pos_act"] = q_act[:, 0]
        saved_dict["v_act"] = vq_act[:, 0]
        saved_dict["acc_act"] = acc_act[:, 0]
        saved_dict["tau"] = tau
        saved_dict["HopParams"] = sqh_p
        print(
            f"Max 1 act: {max1t}, Max 2 act: {max2t}, Reward:{sample_F[i]}, Error vel: {res[1]}")
        save_criterion_traj(robo_urdf_i, path_to_save_result,
                            loop_description_i, joint_description_i, saved_dict)

# load_criterion_traj()
# plt.figure()

# plt.plot(q_act[:, 0])
# plt.plot(list__234[:, 0])
# plt.title("Position")
# plt.xlabel("Time")
# plt.ylabel("Z-Pos")
# plt.legend(["actual vel", "desired vel"])
# plt.grid(True)

# plt.figure()
# plt.plot(acc_act[:, 0])
# plt.plot(list__234[:, 2])
# plt.title("Desired acceleration")
# plt.xlabel("Time")
# plt.ylabel("Z-acc")
# plt.legend(["actual acc", "desired acc"])
# plt.grid(True)

# plt.figure()
# plt.plot(tau[:, 0])
# plt.plot(tau[:, 1])
# plt.title("Actual torques")
# plt.xlabel("Time")
# plt.ylabel("Torques")
# plt.grid(True)

# plt.show()

# path = "results\\th_1909_num1_2024-05-08_19-14-25"

# problem = CalculateCriteriaProblemByWeigths.load(
#     path)  # **{"elementwise_runner":runner})
# checklpoint = load_checkpoint(path)

# optimizer = PymooOptimizer(problem, checklpoint)
# optimizer.load_history(path)

# hist_flat = np.array(optimizer.history["F"]).flatten()
# not_super_best_id = np.argsort(hist_flat)[0]
# sorted_reward = np.sort(hist_flat)
# sorted_reward_big1 = sorted_reward[np.where(sorted_reward < -0.5)]
# # plt.figure()
# # plt.hist(sorted_reward_big1, 10)
# # plt.show()

# best_id = np.argsort(hist_flat)[0]
# best_rew = optimizer.history["F"][best_id]
# not_super_best_rew = optimizer.history["F"][not_super_best_id]
# print(f"Best rew: {best_rew}")
# print(f"Tested rew: {not_super_best_rew}")
# problem.mutate_JP_by_xopt(optimizer.history["X"][not_super_best_id])
# graph = problem.graph

# #actuator = TMotor_AK10_9()
# actuator = MyActuator_RMD_MT_RH_17_100_N()
# thickness = 0.04
# builder = ParametrizedBuilder(
#     DetailedURDFCreatorFixedEE,
#     size_ground=np.array([thickness * 5, thickness * 10, thickness * 2]),
#     actuator=actuator,
#     thickness=thickness)
# robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
#     graph, builder)

# sqh_p = SquatHopParameters(hop_flight_hight=0.25,
#                            squatting_up_hight=0.0,
#                            squatting_down_hight=-0.38,
#                            total_time=0.8)
# hoppa = SimulateSquatHop(sqh_p)

# opti = partial(reward_with_context, hoppa, robo_urdf, joint_description,
#                loop_description)

# res = min_error_control_brute_force(opti)
# x_vec = np.linspace(0.65, 0.9, 10)


# q_act, vq_act, acc_act, tau = hoppa.simulate(robo_urdf,
#                                              joint_description,
#                                              loop_description,
#                                              control_coefficient=res[0],
#                                              is_vis=False)

# trj_f = hoppa.create_traj_equation()
# t = np.linspace(0, sqh_p.total_time, len(q_act))
# list__234 = np.array(list(map(trj_f, t)))

# plt.figure()
# plt.plot(acc_act[:, 0])
# plt.plot(list__234[:, 2])
# plt.title("Desired acceleration")
# plt.xlabel("Time")
# plt.ylabel("Z-acc")
# plt.legend(["actual acc", "desired acc"])
# plt.grid(True)

# plt.figure()
# plt.plot(tau[:, 0])
# plt.plot(tau[:, 1])
# plt.title("Actual torques")
# plt.xlabel("Time")
# plt.ylabel("Torques")
# plt.grid(True)

# plt.figure()

# plt.plot(vq_act[:, 0])
# plt.plot(list__234[:, 1])
# plt.title("Velocities")
# plt.xlabel("Time")
# plt.ylabel("Z-vel")
# plt.legend(["actual vel", "desired vel"])
# plt.grid(True)

# plt.figure()

# plt.plot(q_act[:, 0])
# plt.plot(list__234[:, 0])
# plt.title("Position")
# plt.xlabel("Time")
# plt.ylabel("Z-Pos")
# plt.legend(["actual vel", "desired vel"])
# plt.grid(True)

# plt.show()
# pass
