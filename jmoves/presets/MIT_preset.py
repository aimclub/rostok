import numpy as np

from auto_robot_design.description.builder import (
    ParametrizedBuilder,
    URDFLinkCreater3DConstraints,
    jps_graph2pinocchio_robot_3d_constraints,
    MIT_CHEETAH_PARAMS_DICT,
)


def get_mit_builder():
    thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    density = MIT_CHEETAH_PARAMS_DICT["density"]
    body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]

    builder = ParametrizedBuilder(
        URDFLinkCreater3DConstraints,
        density={"default": density, "G": body_density},
        thickness={"default": thickness, "EE": 0.033},
        actuator={"default": actuator},
        size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
        offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"],
    )

    return builder
