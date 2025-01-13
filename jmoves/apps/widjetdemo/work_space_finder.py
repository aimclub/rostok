from functools import partial
import time
from joblib import Parallel, cpu_count, delayed
import numpy as np


from auto_robot_design.description.builder import jps_graph2pinocchio_robot
from auto_robot_design.optimization.problems import CalculateMultiCriteriaProblem
from auto_robot_design.pinokla.calc_criterion import folow_traj_by_proximal_inv_k_2
from auto_robot_design.description.builder import jps_graph2pinocchio_robot, MIT_CHEETAH_PARAMS_DICT
from apps.widjetdemo import create_reward_manager
from apps.widjetdemo import traj_graph_setup

from auto_robot_design.utils.append_saver import chunk_list, save_result_append
from auto_robot_design.utils.bruteforce import get_n_dim_linspace


def test_graph(problem: CalculateMultiCriteriaProblem, workspace_trj: np.ndarray, x_vec: np.ndarray):
    """Mutate graph by x and check workspace reachability. 

    Args:
        problem (CalculateMultiCriteriaProblem): _description_
        workspace_trj (np.ndarray): _description_
        x_vec (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    problem.mutate_JP_by_xopt(x_vec)
    fixed_robot, free_robot = jps_graph2pinocchio_robot(
        problem.graph, problem.builder)
    poses, q_array, constraint_errors, reach_array = folow_traj_by_proximal_inv_k_2(fixed_robot.model, fixed_robot.data,
                                                                                    fixed_robot.constraint_models, fixed_robot.constraint_data, "EE", workspace_trj)
    return poses, q_array, constraint_errors, reach_array, x_vec


def convert_res_to_dict(poses, q_array, constraint_errors, reach_array, x_vec):
    return {"poses": poses, "q_array": q_array, "constraint_errors": constraint_errors, "reach_array": reach_array, "x_vec": x_vec}


def stack_dicts(dict_list, stack_func=np.stack):
    """
    Stacks dictionaries containing numpy arrays using the specified stacking function.

    Parameters:
    dict_list (list of dict): List of dictionaries to stack.
    stack_func (function): Numpy stacking function to use (e.g., np.vstack, np.hstack, np.concatenate).

    Returns:
    dict: A dictionary with stacked numpy arrays.
    """
    # Initialize an empty dictionary to hold the stacked arrays
    stacked_dict = {}

    # Iterate through the keys of the first dictionary (assuming all dicts have the same keys)
    for key in dict_list[0].keys():
        # Stack the arrays for the current key across all dictionaries
        stacked_dict[key] = stack_func([d[key] for d in dict_list])

    return stacked_dict


def test_chunk(problem: CalculateMultiCriteriaProblem, x_vecs: np.ndarray, workspace_trj: np.ndarray, file_name):
    grabbed_fun = partial(test_graph, problem, workspace_trj)
    parallel_results = []
    cpus = cpu_count(only_physical_cores=True)
    parallel_results = Parallel(cpus, backend="multiprocessing", verbose=100, timeout=60 * 1000)(delayed(grabbed_fun)(i)
                                                                                                 for i in x_vecs)
    list_dict = []
    for i in parallel_results:
        list_dict.append(convert_res_to_dict(*i))
    staced_res = stack_dicts(list_dict)
    save_result_append(file_name, staced_res)
    return staced_res


if __name__ == '__main__':
    start_time = time.time()
    TOPOLGY_NAME = 0
    FILE_NAME = "WORKSPACE_TOP" + str(TOPOLGY_NAME) + ".npz"
    # Needs only for create problem mutate graph by x
    graph, optimizing_joints, constrain_dict, builder, workspace_trajectory = traj_graph_setup.get_graph_and_traj(
        TOPOLGY_NAME)
    reward_manager, crag, soft_constrain = create_reward_manager.get_manager_mock(
        workspace_trajectory)

    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    # Problem needs only for mutate graph by x
    problem = CalculateMultiCriteriaProblem(graph, builder=builder,
                                            jp2limits=optimizing_joints,
                                            crag=[],
                                            soft_constrain=soft_constrain,
                                            rewards_and_trajectories=reward_manager,
                                            Actuator=actuator)

    x_opt, opt_joints, upper_bounds, lower_bounds = problem.convert_joints2x_opt()

    vecs = get_n_dim_linspace(upper_bounds, lower_bounds)
    chunk_vec = list(chunk_list(vecs, 100))
    for num, i_vec in enumerate(chunk_vec):
        try:
            test_chunk(problem, i_vec, workspace_trajectory, FILE_NAME)
        except:
            print("FAILD")
        print(f"Tested chunk {num} / {len(chunk_vec)}")
        ellip = (time.time() - start_time) / 60
        print(f"Remaining minute {ellip}")
