import os
from hashlib import sha256
from pathlib import Path

import numpy as np

from auto_robot_design.motion_planning.trajectory_ik_manager import (
    IK_METHODS, TrajectoryIKManager)
from auto_robot_design.pinokla.calc_criterion import (ComputeInterfaceMoment,
                                                      DataDict,
                                                      along_criteria_calc,
                                                      iterate_over_q_space,
                                                      moment_criteria_calc)
from auto_robot_design.pinokla.loader_tools import (
    Robot, build_model_with_extensions)


def calculate_quasi_static_simdata(free_robot: Robot,
                                   fixed_robot: Robot,
                                   ee_frame_name: str,
                                   traj_6d: np.ndarray,
                                   viz=None, alg_name = "Closed_Loop_PI") -> tuple[DataDict, DataDict]:
    """Calculate criteria for free model(root joint is universal) and 
    fixed model (root joint is weld).

    Args:
        free_robot (Robot): free model
        fixed_robot (Robot): fixed model
        ee_frame_name (str): _description_
        traj_6d (np.ndarray): Desired end-effector trajectory
        viz (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple[DataDict, DataDict]: free data, closed data
    """
    # create the trajectory manager and set the solver
    ik_manager = TrajectoryIKManager()
    ik_manager.register_model(fixed_robot.model, fixed_robot.constraint_models)
    ik_manager.set_solver(alg_name)
    poses, q_fixed, constraint_errors,reach_array = ik_manager.follow_trajectory(traj_6d)

    # add standard body position to all points in the q space
    normal_pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    free_body_q = np.repeat(normal_pose[np.newaxis, :], len(q_fixed), axis=0)
    free_space_q = np.concatenate((free_body_q, q_fixed), axis=1)
    # perform calculations of the Jacobians, inertial and dq for free and fixed robots
    res_dict_free = iterate_over_q_space(free_robot, free_space_q,
                                         ee_frame_name)
    res_dict_fixed = iterate_over_q_space(fixed_robot, q_fixed, ee_frame_name)
    # add trajectory following characteristics to the result dictionaries
    res_dict_fixed["traj_6d_ee"] = poses
    res_dict_free["traj_6d_ee"] = poses

    res_dict_fixed["traj_6d"] = traj_6d
    res_dict_free["traj_6d"] = traj_6d

    res_dict_fixed["error"] = constraint_errors
    res_dict_free["error"] = constraint_errors

    res_dict_fixed["is_reach"] = reach_array
    res_dict_free["is_reach"] = reach_array

    res_dict_fixed['q'] = q_fixed
    res_dict_free['q'] = free_space_q

    return res_dict_free, res_dict_fixed


class CriteriaAggregator:
    """Create models from urdf and calculate criteria for the given trajectory.
    """

    def __init__(self, dict_moment_criteria: dict[str, ComputeInterfaceMoment],
                 dict_along_criteria: dict[str, ComputeInterfaceMoment], alg_name="Closed_Loop_PI") -> None:
        self.dict_moment_criteria = dict_moment_criteria
        self.dict_along_criteria = dict_along_criteria
        self.end_effector_name = "EE"
        self.IK_alg_name = alg_name

    def get_criteria_data(self, fixed_robot, free_robot, traj_6d, n_auxiliary_points:int = 50, viz=None):
        """Perform calculating

        Args:
            urdf_str (str): _description_
            mot_des (dict): _description_
            loop_des (dict): _description_

        Returns:
            dict: data calculated for each trajectory point
            dict: data calculated as a result of the whole simulation
            dict: results of trajectory following for the fixed robot 
        """

        # perform calculations of the data required to calculate the fancy mech criteria
        res_dict_free, res_dict_fixed = calculate_quasi_static_simdata(
            free_robot, fixed_robot, self.end_effector_name, traj_6d,viz=viz, alg_name=self.IK_alg_name)
        # calculate the criteria that can be assigned to each point at the trajectory 
        point_criteria_vector = moment_criteria_calc(self.dict_moment_criteria,
                                                  res_dict_free, res_dict_fixed)
        # calculate criteria that characterize the performance along the whole trajectory
        trajectory_criteria = along_criteria_calc(self.dict_along_criteria,res_dict_free,
                                                res_dict_fixed, fixed_robot, free_robot)
        # remove the first 50 points from the results, as they belong to the auxiliary part of the trajectory
        for d in [point_criteria_vector,  res_dict_fixed]:
            for k,v in d.items():
                d[k] = v[n_auxiliary_points::]
        return point_criteria_vector, trajectory_criteria, res_dict_fixed


def save_criterion_traj(
    urdf: str,
    directory: str,
    loop_description: dict,
    mot_description: dict,
    data_dict: dict,
):

    graph_name = sha256(urdf.encode()).hexdigest()
    path_with_name = Path(directory) / graph_name
    savable_dict = {
        "urdf": urdf,
        "loop_description": loop_description,
        "mot_description": mot_description,
    }

    savable_dict.update(data_dict)
    os.makedirs(Path(directory), exist_ok=True)
    np.savez(path_with_name, **savable_dict)


def load_criterion_traj(name: str):
    path = Path(name)
    load_data = np.load(path, allow_pickle=True)
    return dict(load_data)
