
import time
from collections import UserDict
from enum import IntFlag, auto
from typing import NamedTuple, Optional

import numpy as np
import pinocchio as pin
from numpy.linalg import norm
from auto_robot_design.pinokla.closed_loop_jacobian import (
    constraint_jacobian_active_to_passive,
)
from auto_robot_design.pinokla.closed_loop_jacobian import (
    closedLoopInverseKinematicsProximal,
    ConstraintFrameJacobian)
from auto_robot_design.pinokla.closed_loop_kinematics import (
    ForwardK, closedLoopProximalMount)
from auto_robot_design.pinokla.criterion_math import (calc_manipulability,
                                                      ImfProjections, calc_actuated_mass, calc_effective_inertia,
                                                      calc_force_ell_projection_along_trj, calc_IMF, calculate_mass,
                                                      convert_full_J_to_planar_xz)
from auto_robot_design.pinokla.loader_tools import Robot


class MovmentSurface(IntFlag):
    XZ = auto()
    ZY = auto()
    YX = auto()


class PsedoStepResault(NamedTuple):
    J_closed: np.ndarray = None
    M: np.ndarray = None
    dq: np.ndarray = None


class DataDict(UserDict):
    """Dict to store simulation data. Each value is np.array with same size.

    Args:
        UserDict (_type_): _description_
    """

    def get_frame(self, index):
        """Get values with same index.

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        extracted_elements = {}
        for key, array in self.items():
            extracted_elements[key] = array[index]
        return extracted_elements

    def get_data_len(self):
        return len(self[next(iter(self))])


def search_workspace(
    model,
    data,
    effector_frame_name: str,
    base_frame_name: str,
    q_space: np.ndarray,
    actuation_model,
    constraint_models,
    viz=None,
):
    """Iterate forward kinematics over q_space and try to minimize constrain value.

    Args:
        model (_type_): _description_
        data (_type_): _description_
        effector_frame_name (str): _description_
        base_frame_name (str): _description_
        q_space (np.ndarray): _description_
        actuation_model (_type_): _description_
        constraint_models (_type_): _description_
        viz (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    c = 0
    q_start = pin.neutral(model)
    workspace_xyz = np.empty((len(q_space), 3))
    available_q = np.empty((len(q_space), len(q_start)))
    for q_sample in q_space:

        q_dict_mot = zip(actuation_model.idqmot, q_sample)
        for key, value in q_dict_mot:
            q_start[key] = value
        q3, error = ForwardK(
            model,
            constraint_models,
            actuation_model,
            q_start,
            150,
        )

        if error < 1e-11:
            if viz:
                viz.display(q3)
                time.sleep(0.005)
            q_start = q3
            pin.framesForwardKinematics(model, data, q3)
            id_effector = model.getFrameId(effector_frame_name)
            id_base = model.getFrameId(base_frame_name)
            effector_pos = data.oMf[id_effector].translation
            base_pos = data.oMf[id_base].translation
            transformed_pos = effector_pos - base_pos

            workspace_xyz[c] = transformed_pos
            available_q[c] = q3
            c += 1
    return (workspace_xyz[0:c], available_q[0:c])


def folow_traj_by_proximal_inv_k(
    model,
    data,
    constraint_models,
    constraint_data,
    end_effector_frame: str,
    traj_6d: np.ndarray,
    viz=None,
    q_start: np.ndarray = None,
):
    """Solve the inverse kinematic problem

    Args:
        model (_type_): _description_
        data (_type_): _description_
        constraint_models (_type_): _description_
        constraint_data (_type_): _description_
        end_effector_frame (str): _description_
        traj_6d (np.ndarray): _description_
        viz (_type_, optional): _description_. Defaults to None.
        q_start (np.ndarray, optional): _description_. Defaults to None.

    Returns:
        np.array: end-effector positions in final state
        np.array: joint coordinates in final state
        np.array: deviations from the desired position

    """
    if q_start:
        q = q_start
    else:
        q = pin.neutral(model)

    ee_frame_id = model.getFrameId(end_effector_frame)
    poses = np.zeros((len(traj_6d), 3))
    q_array = np.zeros((len(traj_6d), len(q)))
    constraint_errors = np.zeros((len(traj_6d), 1))

    for num, i_pos in enumerate(traj_6d):
        q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
            model,
            data,
            constraint_models,
            constraint_data,
            i_pos,
            ee_frame_id,
            onlytranslation=True,
            q_start=q,
        )
        if not is_reach:
            q = closedLoopProximalMount(
                model, data, constraint_models, constraint_data, q
            )
        if viz:
            viz.display(q)
            time.sleep(0.1)

        pin.framesForwardKinematics(model, data, q)
        poses[num] = data.oMf[ee_frame_id].translation
        q_array[num] = q
        constraint_errors[num] = min_feas

    return poses, q_array, constraint_errors


def folow_traj_by_proximal_inv_k_2(
    model,
    data,
    constraint_models,
    constraint_data,
    end_effector_frame: str,
    traj_6d: np.ndarray,
    viz=None,
    q_start: np.ndarray = None,
):
    """Solve the inverse kinematic problem

    Args:
        model (_type_): _description_
        data (_type_): _description_
        constraint_models (_type_): _description_
        constraint_data (_type_): _description_
        end_effector_frame (str): _description_
        traj_6d (np.ndarray): _description_
        viz (_type_, optional): _description_. Defaults to None.
        q_start (np.ndarray, optional): _description_. Defaults to None.

    Returns:
        np.array: end-effector positions in final state
        np.array: joint coordinates in final state
        np.array: deviations from the desired position

    """
    if q_start:
        q = q_start
    else:
        q = pin.neutral(model)

    ee_frame_id = model.getFrameId(end_effector_frame)
    poses = np.zeros((len(traj_6d), 3))
    reach_array = np.zeros(len(traj_6d))
    q_array = np.zeros((len(traj_6d), len(q)))
    constraint_errors = np.zeros((len(traj_6d), 1))

    for num, i_pos in enumerate(traj_6d):
        q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
            model,
            data,
            constraint_models,
            constraint_data,
            i_pos,
            ee_frame_id,
            onlytranslation=True,
            q_start=q,
        )
        if not is_reach:
            q = closedLoopProximalMount(
                model, data, constraint_models, constraint_data, q
            )
        if viz:
            viz.display(q)
            time.sleep(0.1)

        pin.framesForwardKinematics(model, data, q)
        poses[num] = data.oMf[ee_frame_id].translation
        q_array[num] = q
        constraint_errors[num] = min_feas
        reach_array[num] = is_reach

    return poses, q_array, constraint_errors, reach_array


def pseudo_static_step(robot: Robot, q_state: np.ndarray,
                       ee_frame_name: str) -> PsedoStepResault:

    ee_frame_id = robot.model.getFrameId(ee_frame_name)
    pin.framesForwardKinematics(robot.model, robot.data, q_state)
    pin.computeJointJacobians(robot.model, robot.data, q_state)
    pin.centerOfMass(robot.model, robot.data, q_state)

    # J_closed = ConstraintFrameJacobian(
    #     robot.model,
    #     robot.data,
    #     robot.constraint_models,
    #     robot.constraint_data,
    #     robot.actuation_model,
    #     q_state,
    #     ee_frame_id,
    #     robot.data.oMf[ee_frame_id].action @ np.zeros(6),
    # )
    _dq_dqmot, __ = constraint_jacobian_active_to_passive(
        robot.model,
        robot.data,
        robot.constraint_models,
        robot.constraint_data,
        robot.actuation_model,
        q_state,
    )

    J_closed = (
        pin.computeFrameJacobian(
            robot.model, robot.data, q_state, ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        @ _dq_dqmot
    )
    #[[0,2]]
    LJ = []
    for cm, cd in zip(robot.constraint_models, robot.constraint_data):
        Jc = pin.getConstraintJacobian(robot.model, robot.data, cm, cd)
        LJ.append(Jc)

    M = pin.crba(robot.model, robot.data, q_state)
    # TODO: force Kirill to explain what is this and why we need it
    #dq = dq_dqmot(robot.model, robot.actuation_model, LJ)
    dq =_dq_dqmot
    return PsedoStepResault(J_closed, M, dq)


def iterate_over_q_space(robot: Robot, q_space: np.ndarray,
                         ee_frame_name: str):
    zero_step = pseudo_static_step(robot, q_space[0], ee_frame_name)

    res_dict = DataDict()
    for key, value in zero_step._asdict().items():
        alocate_array = np.zeros(
            (len(q_space), *value.shape), dtype=np.float64)
        res_dict[key] = alocate_array

    for num, q_state in enumerate(q_space):
        one_step_res = pseudo_static_step(robot, q_state, ee_frame_name)
        for key, value in one_step_res._asdict().items():
            res_dict[key][num] = value

    return res_dict


class ComputeInterfaceMoment:
    """Abstract class for calculate criterion on each step of simulation."""

    def __init__(self) -> None:
        """Determine what type of data is needed for the calculation. 
        From an free model or fixed base model 
        """
        self.is_fixed = True

    def __call__(
        self, data_frame: dict[str, np.ndarray], robo: Robot = None
    ) -> np.ndarray:
        """Call on every data frame from data_dict.

        Args:
            data_frame (dict[str, np.ndarray]): see get_frame
            robo (Robot, optional): model description. Defaults to None.

        Raises:
            NotImplemented: _description_

        Returns:
            np.ndarray: _description_
        """
        raise NotImplemented

    def output_matrix_shape(self) -> Optional[tuple]:
        return None


class ComputeInterface:
    """Abstract class for calculate criterion on data trajectory of simulation."""

    def __init__(self) -> None:
        """Determine what type of data is needed for the calculation. 
        From an free model or fixed base model 
        """
        self.is_fixed = True

    def __call__(self, data_dict: DataDict, robo: Robot = None):
        """Call on output data_dict, that contain whole simulation data. See iterate_over_q_space and pseudo_static_step.

        Args:
            data_dict (DataDict): simulation data dict
            robo (Robot, optional): model description. Defaults to None.

        Raises:
            NotImplemented: _description_
        """
        raise NotImplemented


class ImfCompute(ComputeInterfaceMoment):
    """Wrapper for IMF. Criterion implementation src is criterion_math"""

    def __init__(self, projection: ImfProjections) -> None:
        self.projection = projection
        self.is_fixed = False

    def __call__(
        self, data_frame: dict[str, np.ndarray], robo: Robot = None
    ) -> np.ndarray:
        imf = calc_IMF(
            data_frame["M"], data_frame["dq"], data_frame["J_closed"], self.projection
        )
        return imf


class EffectiveInertiaCompute(ComputeInterfaceMoment):
    """Wrapper for Effective Inertia. Criterion implementation src is criterion_math"""

    def __init__(self) -> None:
        self.is_fixed = True

    def __call__(
        self, data_frame: dict[str, np.ndarray], robo: Robot = None
    ) -> np.ndarray:
        eff_inertia = calc_effective_inertia(
            data_frame["M"], data_frame["dq"], data_frame["J_closed"]
        )
        return eff_inertia


class ActuatedMass(ComputeInterfaceMoment):
    """Wrapper for Actuated_Mass. Criterion implementation src is criterion_math"""

    def __init__(self) -> None:
        self.is_fixed = True

    def __call__(
        self, data_frame: dict[str, np.ndarray], robo: Robot = None
    ) -> np.ndarray:
        eff_inertia = calc_actuated_mass(
            data_frame["M"], data_frame["dq"], data_frame["J_closed"]
        )
        return eff_inertia


class ManipCompute(ComputeInterfaceMoment):
    """Wrapper for manipulability. Criterion implementation src is criterion_math"""

    def __init__(self, surface: MovmentSurface) -> None:
        self.surface = surface
        self.is_fixed = True

    def __call__(
        self, data_frame: dict[str, np.ndarray], robo: Robot = None
    ) -> np.ndarray:
        if self.surface == MovmentSurface.XZ:
            target_J = data_frame["J_closed"]
            target_J = convert_full_J_to_planar_xz(target_J)
            target_J = target_J[:2, :2]
        else:
            raise NotImplemented
        manip_space = calc_manipulability(target_J)
        return manip_space


class ManipJacobian(ComputeInterfaceMoment):
    """Wrapper for manipulability. Criterion implementation src is criterion_math"""

    def __init__(self, surface: MovmentSurface) -> None:
        self.surface = surface
        self.is_fixed = True

    def __call__(
        self, data_frame: dict[str, np.ndarray], robo: Robot = None
    ) -> np.ndarray:
        if self.surface == MovmentSurface.XZ:
            target_J = data_frame["J_closed"]
            target_J = convert_full_J_to_planar_xz(target_J)
            target_J = target_J[:2, :2]
        else:
            raise NotImplemented

        return target_J


class ForceCapabilityProjectionCompute(ComputeInterface):
    """Wrapper for calculate projection force ellipsoid axis to ez and xz trajectory. Criterion implementation src is criterion_math
    Return sum of absolute dot product of trajectory and u1, u2 (axis force ellips), z-axis and u1, u2 (axis force ellips).

    Fucntion Call Returns:
        tuple: A tuple containing the following values:
            - abs_dot_product_traj_u1: The absolute dot product of the trajectory and u1.
            - abs_dot_product_traj_u2: The absolute dot product of the trajectory and u2.
            - abs_dot_product_z_u1: The absolute dot product of the z-axis and u1.
            - abs_dot_product_z_u2: The absolute dot product of the z-axis and u2.
    """

    def __init__(self) -> None:
        self.is_fixed = True

    def __call__(self, data_dict: DataDict, robo: Robot = None) -> np.ndarray:
        """
        Calculate projection force ellipsoid axis to ez and xz trajectory.

        Args:
            data_dict (DataDict): A dictionary containing the required data for calculation.
            robo (Robot, optional): The robot object. Defaults to None.

        Returns:
            tuple: A tuple containing the following values:
                - abs_dot_product_traj_u1: The absolute dot product of the trajectory and u1.
                - abs_dot_product_traj_u2: The absolute dot product of the trajectory and u2.
                - abs_dot_product_z_u1: The absolute dot product of the z-axis and u1.
                - abs_dot_product_z_u2: The absolute dot product of the z-axis and u2.
        """

        Jc_traj = data_dict["J_closed"]
        traj_xz = data_dict["traj_6d_ee"][:, [0, 2]]

        Jc_xz_traj = Jc_traj[:, [0, 2], :]
        # U, S, Vh = np.linalg.svd(Jc_xz_traj, hermitian=True)
        U, S, Vh = np.linalg.svd(Jc_xz_traj)
        USsec = np.array([np.diag(1/s) @ u for u, s in zip(U, S)])
        US1 = USsec[:, 0, :]
        US2 = USsec[:, 1, :]

        d_traj_xz = np.diff(traj_xz, axis=0)
        d_traj_xz = np.vstack([d_traj_xz, [0, 0]])

        abs_dot_product_traj_u1 = np.sum(
            np.abs(np.sum(US1 * d_traj_xz, axis=1))).squeeze()
        abs_dot_product_traj_u2 = np.sum(
            np.abs(np.sum(US2 * d_traj_xz, axis=1))).squeeze()
        abs_dot_product_z_u1 = np.sum(np.abs(US1[:, 1])).squeeze()
        abs_dot_product_z_u2 = np.sum(np.abs(US2[:, 1])).squeeze()

        return (
            abs_dot_product_traj_u1,
            abs_dot_product_traj_u2,
            abs_dot_product_z_u1,
            abs_dot_product_z_u2,
        )


class NeutralPoseMass(ComputeInterface):
    """Wrapper for calculate total mass of robot. Criterion implementation src is criterion_math"""

    def __init__(self) -> None:
        self.is_fixed = False

    def __call__(self, data_dict: DataDict, robo: Robot = None):
        return calculate_mass(robo)


class ForceEllProjections(ComputeInterface):
    """Wrapper for calc_force_ell_projection_along_trj.

    Args:
        ComputeInterface (_type_): _description_
    """

    def __init__(self) -> None:
        self.is_fixed = True

    def __call__(self, data_dict: DataDict, robo: Robot = None):
        ell_params = calc_force_ell_projection_along_trj(
            data_dict["J_closed"], data_dict["traj_6d"]
        )
        return ell_params


class TranslationErrorMSE(ComputeInterface):
    """Calculate mean square error for translation part of end effector trajectory"""

    def __init__(self) -> None:
        self.is_fixed = True

    def __call__(self, data_dict: DataDict, robo: Robot = None):

        errors = norm(data_dict["traj_6d"][:, :3] - data_dict["traj_6d_ee"][:, :3],
                      axis=1)
        mean_error = np.mean(errors)
        if mean_error < 1e-6:
            return 0
        return mean_error


def moment_criteria_calc(calculate_desription: dict[str,
                                                    ComputeInterfaceMoment],
                         data_dict_free: DataDict, data_dict_fixed: DataDict,
                         robo: Robot = None) -> DataDict:
    """Calculate all critrion from calculate_desription. Each criterion is 
    called on data frames that represent the data at each point in time.

    Args:
        calculate_desription (dict[str, ComputeInterfaceMoment]): key is criterion name, value is critrion resault
        data_dict (DataDict): _description_
        robo (Robot, optional): _description_. Defaults to None.

    Returns:
        DataDict: _description_
    """
    res_dict = DataDict()
    for key, criteria in calculate_desription.items():
        if criteria.is_fixed:
            data_dict = data_dict_fixed
        else:
            data_dict = data_dict_free

        shape = criteria.output_matrix_shape()
        if shape:
            res_dict[key] = np.zeros(
                (data_dict.get_data_len(), *shape), dtype=np.float32
            )
        else:
            frame_data = data_dict.get_frame(0)
            zero_step = criteria(frame_data)
            res_dict[key] = np.zeros(
                (data_dict.get_data_len(), *zero_step.shape), dtype=np.float32
            )
            # Need implement alocate from zero step data size
            # raise NotImplemented

    for index in range(data_dict_fixed.get_data_len()):
        for key, criteria in calculate_desription.items():
            if criteria.is_fixed:
                data_dict = data_dict_fixed
            else:
                data_dict = data_dict_free

            data_frame = data_dict.get_frame(index)
            res_dict[key][index] = criteria(data_frame, robo)
    return res_dict


def along_criteria_calc(calculate_desription: dict[str, ComputeInterface],
                        data_dict_free: DataDict, data_dict_fixed: DataDict,
                        robo_fixed: Robot = None, robo_free: Robot = None) -> dict:
    """Each criterion get the entire DataDict and Robot.

    Args:
        calculate_desription (dict[str, ComputeInterface]): _description_
        data_dict (DataDict): _description_
        robo (Robot, optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """
    res_dict = {}
    for key, criteria in calculate_desription.items():
        if criteria.is_fixed:
            data_dict = data_dict_fixed
            robo = robo_fixed
        else:
            data_dict = data_dict_free
            robo = robo_free
        res_dict[key] = criteria(data_dict, robo)
    return res_dict
