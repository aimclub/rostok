from typing import Any, Dict, List, Tuple

import pychrono.core as chrono

from rostok.block_builder_chrono.block_classes import (ChronoRevolveJoint, JointInputTypeChrono)


class RobotControllerChrono:

    def __init__(self, joint_vector, parameters: Dict[int, Any]):
        self.joints: List[Tuple[int, ChronoRevolveJoint]] = joint_vector
        self.initialize_functions(parameters)

    def initialize_functions(self, parameters):
        if len(parameters) != len(self.joints):
            raise Exception("some joints are not parametrized")

        for i, joint in enumerate(self.joints):
            chr_function = chrono.ChFunction_Const(float(parameters[i]))
            joint[1].joint.SetTorqueFunction(chr_function)

    def update_functions(self, robot_data, environment_data):
        pass
