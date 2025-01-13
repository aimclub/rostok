import numpy as np
import numpy.linalg as la

import pinocchio as pin

from auto_robot_design.pinokla.loader_tools import build_model_with_extensions


def calculate_mass(urdf, joint_description, loop_description):
    free_robo = build_model_with_extensions(
        urdf, joint_description, loop_description, False
    )
    pin.computeAllTerms(
        free_robo.model,
        free_robo.data,
        np.zeros(free_robo.model.nq),
        np.zeros(free_robo.model.nv),
    )
    total_mass = pin.computeTotalMass(free_robo.model, free_robo.data)
    com_dist = la.norm(pin.centerOfMass(free_robo.model, free_robo.data))

    return total_mass 
