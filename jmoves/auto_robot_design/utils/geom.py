import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R
import modern_robotics as mr


def calculate_rot_vec2_to_vec1(vec1: np.ndarray,
                            vec2: np.ndarray = np.array([0, 0, 1])):
    """Calculate transformation from `vec2` to vector `vec1`

    Args:
        p1 (np.ndarray): point of vector's start
        p2 (np.ndarray): point of vector's end
        vec (np.ndarray, optional): Vector transform from. Defaults to np.array([0, 0, 1]).

    Returns:
        tuple: position: np.ndarray, rotation: scipy.spatial.rotation, length: float
    """
    angle = np.arccos(np.inner(vec2, vec1) / la.norm(vec1) / la.norm(vec2))
    axis = mr.VecToso3(vec2) @ vec1
    if not np.isclose(np.sum(axis), 0):
        axis /= la.norm(axis)

    rot = R.from_rotvec(axis * angle)
    
    return rot

def calculate_transform_with_2points(p1: np.ndarray, 
                                     p2: np.ndarray,
                                     vec: np.ndarray = np.array([0, 0, 1])):
    """Calculate transformation from `vec` to vector build with points `p1` and `p2`

    Args:
        p1 (np.ndarray): point of vector's start
        p2 (np.ndarray): point of vector's end
        vec (np.ndarray, optional): Vector transform from. Defaults to np.array([0, 0, 1]).

    Returns:
        tuple: position: np.ndarray, rotation: scipy.spatial.rotation, length: float
    """
    v_l = p2 - p1
    angle = np.arccos(np.inner(vec, v_l) / la.norm(v_l) / la.norm(vec))
    axis = mr.VecToso3(vec[:3]) @ v_l[:3]
    if not np.isclose(np.sum(axis), 0):
        axis /= la.norm(axis)

    rot = R.from_rotvec(axis * angle)
    pos = (p2 + p1) / 2
    length = la.norm(v_l)
    
    return pos, rot, length