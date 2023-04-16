from typing import List, Tuple, Dict, Any

import pychrono.core as chrono
from rostok.block_builder_chrono.block_classes import JointInputTypeChrono

class ControlFunctionBuilder:
    def __init__(self, parameters):
        self.parameters = parameters
    def create_function(self, joint):
        pass

class ControlFunctionChrono(chrono.ChFunction):

    def __init__(self):
        self.function = None

    def update_function(self, robot_info, env_info):
        pass

    def initialize(self):
        pass



class RobotControllerChrono:
    def __init__(self, joint_vector, parameters: Dict[int, Any]):
        self.joint_functions:List[ControlFunctionChrono] = []
        self.joints = joint_vector
        self.initialize_functions(parameters)

    def initialize_functions(self, parameters):
        for joint in self.joints:
            

    def bind_functions(self):
        
        self.type_variants = {
            JointInputTypeChrono.TORQUE: lambda x: self.get_joint().SetTorqueFunction(x),
            JointInputTypeChrono.VELOCITY: lambda x: self.get_joint().SetSpeedFunction(x),
            JointInputTypeChrono.POSITION: lambda x: self.get_joint().SetAngleFunction(x),
            JointInputTypeChrono.UNCONTROL: None
        }

    def update_functions(self, robot_data, environment_data):
        for function in self.joint_functions:
            function.update_function(robot_data, environment_data)
