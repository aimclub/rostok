from enum import IntFlag, auto

from matplotlib.pylab import LinAlgError
import numpy as np
import pinocchio as pin

from auto_robot_design.pinokla.loader_tools import Robot

import numpy.linalg as la


class ImfProjections(IntFlag):
    X = auto()
    Y = auto()
    Z = auto()
    ALL = auto()


def convert_full_J_to_planar_xz(full_J: np.ndarray):
    ret = np.row_stack((full_J[0], full_J[2], full_J[4]))
    return ret

def convert_full_J_to_planar_xy(full_J: np.ndarray):
    ret = np.row_stack((full_J[0], full_J[1], full_J[3]))
    return ret

def calc_manipulability(jacob: np.ndarray):
    U, S, Vh = np.linalg.svd(jacob)
    return np.prod(S)


def calc_svd_jacobian(jacob: np.ndarray):
    U, S, Vh = np.linalg.svd(jacob)
    return U, S, Vh


def calc_IMF(M: np.ndarray,
             dq: np.ndarray,
             J_closed: np.ndarray,
             projection: ImfProjections = ImfProjections.Z):

    x = np.array([1, 0, 0, 0, 0, 0])
    y = np.array([0, 1, 0, 0, 0, 0])
    z = np.array([0, 0, 1, 0, 0, 0])


    Mmot_free = dq.T @ M @ dq
    Lambda_free = np.linalg.inv(
        J_closed @ np.linalg.inv(Mmot_free) @ J_closed.T)
    Lambda_free_lock = np.linalg.inv(J_closed[:6, :6] @ np.linalg.inv(
        Mmot_free[:6, :6]) @ J_closed[:6, :6].T)

    if projection == ImfProjections.X:
        ret_IMF = 1 - (x.T @ Lambda_free @ x) / (x.T @ Lambda_free_lock @ x)
    elif projection == ImfProjections.Y:
        ret_IMF = 1 - (y.T @ Lambda_free @ y) / (y.T @ Lambda_free_lock @ y)
    elif projection == ImfProjections.Z:
        ret_IMF = 1 - (z.T @ Lambda_free @ z) / (z.T @ Lambda_free_lock @ z)
    elif projection == ImfProjections.ALL:
        ret_IMF = np.linalg.det(
            np.identity(6) - Lambda_free @ np.linalg.inv(Lambda_free_lock))

    return ret_IMF

def calc_effective_inertia(M: np.ndarray,
             dq: np.ndarray,
             J_closed: np.ndarray,
             projection: ImfProjections = ImfProjections.Z):

    Mmot = dq.T @ M @ dq
    J_closed = J_closed[[0,2]]
    Lambda = np.linalg.inv(
        J_closed @ np.linalg.inv(Mmot) @ J_closed.T)

    return Lambda

def calc_actuated_mass(M: np.ndarray,
             dq: np.ndarray,
             J_closed: np.ndarray,
             projection: ImfProjections = ImfProjections.Z):

    Mmot = dq.T @ M @ dq
    Mmot = Mmot[:2, :2]
    return Mmot


def calc_force_ellips_space(jacob: np.ndarray):
    try:
        ret1 = np.linalg.det(np.linalg.inv(jacob).T @ np.linalg.inv(jacob))
    except LinAlgError:
        ret1 = 0
    return ret1


def calc_svd_j_along_trj_trans(traj_J_closed):
    array_manip = []
    for num, J in enumerate(traj_J_closed):
        planar_J = convert_full_J_to_planar_xz(J)
        trans_planar_J = planar_J[:2, :2]
        # svd_j = calc_svd_jacobian(trans_planar_J)
        array_manip.append(trans_planar_J)

    return array_manip


def calc_force_ell_projection_along_trj(traj_J_closed, traj_6d):
    """
    Calculates the force ellipsoid projection along a trajectory.

    Args:
        traj_J_closed (numpy.ndarray): Closed trajectory of Jacobian matrices.
        traj_6d (numpy.ndarray): 6-dimensional trajectory.

    Returns:
        dict: Dictionary containing the following keys:
            - "u1_traj": Absolute dot product of trajectory and u1.
            - "u2_traj": Absolute dot product of trajectory and u2.
            - "u1_z": Absolute dot product of z-axis and u1.
            - "u2_z": Absolute dot product of z-axis and u2.
    """
    svd_J = calc_svd_j_along_trj_trans(traj_J_closed)

    d_xy = np.diff(traj_6d[:, np.array([0, 2])], axis=0)
    d_xy = np.vstack([d_xy, [0, 0]])
    traj_j_svd = [np.linalg.svd(J_ck) for J_ck in svd_J]

    u1 = np.array([1 / J_svd[1][0] * J_svd[0][0, :] for J_svd in traj_j_svd])
    u2 = np.array([1 / J_svd[1][1] * J_svd[0][1, :] for J_svd in traj_j_svd])

    abs_dot_product_traj_u1 = np.abs(np.sum(u1 * d_xy, axis=1).squeeze())
    abs_dot_product_traj_u2 = np.abs(np.sum(u2 * d_xy, axis=1).squeeze())

    abs_dot_product_z_u1 = u1[:, 1]
    abs_dot_product_z_u2 = u2[:, 1]

    out = {
        "u1_traj": abs_dot_product_traj_u1,
        "u2_traj": abs_dot_product_traj_u2,
        "u1_z": abs_dot_product_z_u1,
        "u2_z": abs_dot_product_z_u2
    }
    return out


def calculate_mass(robo: Robot):
    q_0 = pin.neutral(robo.model)
    pin.computeAllTerms(
        robo.model,
        robo.data,
        q_0,
        np.zeros(robo.model.nv),
    )
    total_mass = pin.computeTotalMass(robo.model, robo.data)
    com_dist = la.norm(pin.centerOfMass(robo.model, robo.data))

    return total_mass
