from math import sin
from typing import Any, Dict, List, Tuple

import pychrono.core as chrono

from rostok.block_builder_chrono.block_classes import (ChronoRevolveJoint,
                                                       JointInputTypeChrono)
from rostok.virtual_experiment.sensors import Sensor

class MyConst(chrono.ChFunction_Const):
    def __init__(self, c):
        print("My const init")
        super().__init__(c)

    def Get_y(self, x):
        print(f"Get_y of the const called with: {x}")
        super().Get_y(x)

    def Get_yconst(self):
        print(f"Get_yconst of the const called with: {x}")
        super().Get_yconst(x)

    def Get_y_dx(self, x):
        print(f"Get_y_dx of the const called with: {x}")
        super().Get_y_dx(x)

class RobotControllerChrono:
    """General controller. Any controller should be subclass of this class.
    
        Attributes:
            joints (List[Tuple[int, ChronoRevolveJoint]]): list of all joints in the mechanism
            parameters: vector of parameters for joints
            trajectories: trajectories for the joints
            functions: list of functions currently attached to joints
    """
    def __init__(self, joint_vector, parameters: Dict[int, Any], trajectories=None):
        """Initialize class fields and call the initialize_functions() to set starting state"""
        self.joints: List[Tuple[int, ChronoRevolveJoint]] = joint_vector
        self.parameters = parameters
        self.trajectories = trajectories
        self.functions = []
        self.initialize_functions()

    def get_joint_by_id(self, idx: int):
        """Returns joint object by its id in the graph."""
        for joint in self.joints:
            if joint[0] == idx:
                return joint[1]
        return None

    def initialize_functions(self):
        """Attach initial functions to the joints."""
        if len(self.parameters) != len(self.joints):
            raise Exception("Parameter list should have the length of joint list!")

    def update_functions(self, time, robot_data, environment_data):
        pass

class ConstController(RobotControllerChrono):

    def initialize_functions(self):
        super().initialize_functions()
        for i, joint in enumerate(self.joints):
            chr_function = chrono.ChFunction_Const(float(self.parameters[i]))
            joint[1].joint.SetTorqueFunction(chr_function)
            self.functions.append(chr_function)

    def update_functions(self, time, robot_data, environment_data):
        pass


class SinControllerChronoFn(RobotControllerChrono):
    """Controller that set sinusoidal torques for all joints."""
    def initialize_functions(self):
        super().initialize_functions()
        parameters = self.parameters
        for i, joint in enumerate(self.joints):
            chr_function = chrono.ChFunction_Sine(0.0, parameters[i][1] / 6.28, parameters[i][0])
            joint[1].joint.SetTorqueFunction(chr_function)
            self.functions.append(chr_function)

    def update_functions(self, time, robot_data, environment_data):
        pass


class SinControllerChrono(RobotControllerChrono):
    """Controller that sets sinusoidal torques using constant update at each step."""
    def initialize_functions(self):
        super().initialize_functions()
        for _, joint in enumerate(self.joints):
            f = MyConst(0.0)
            joint[1].joint.SetTorqueFunction(f)
            self.functions.append(f)

    def update_functions(self, time, robot_data, environment_data):
        for i, _ in enumerate(self.joints):
            current_const = self.parameters[i][0] * sin(self.parameters[i][1] * time)
            self.functions[i].Set_yconst(current_const)


class ConstReverseControllerChrono(RobotControllerChrono):

    def __init__(self, joint_vector, parameters: Dict[int, Any], trajectories=None):
        super().__init__(joint_vector, parameters, trajectories=None)
        self.change_prev_step = False
        self.function_list = []

    def initialize_functions(self):
        if len(self.parameters) != len(self.joints):
            raise Exception("some joints are not parametrized")
        for i, joint in enumerate(self.joints):
            chr_function = chrono.ChFunction_Const(float(self.parameters[i]))
            joint[1].joint.SetTorqueFunction(chr_function)
            self.function_list.append(chr_function)

    def update_functions(self, time, robot_data: Sensor, environment_data):
        i = 0
        for item in robot_data.joint_body_map.items():
            if not robot_data.amount_contact_forces(
                    item[1][1]) is None and not self.change_prev_step:
                joint: ChronoRevolveJoint = self.get_joint_by_id(item[0])
                current_const = joint.joint.GetTorqueFunction().Get_y(0)
                
                #joint.joint.SetTorqueFunction(chrono.ChFunction_Const(-current_const))
                #joint.joint.GetTorqueFunction
                self.change_prev_step = True
                return

        self.change_prev_step = False


class RobotControllerTorqueTrajectoryChrono(RobotControllerChrono):

    def __init__(self, joint_vector, parameters: Dict[int, Any], trajectories=None):
        super().__init__(joint_vector, parameters, trajectories)

    def initialize_functions(self):
        if len(self.parameters) != len(self.joints):
            raise Exception("some joints are not parametrized")

        for i, joint in enumerate(self.joints):
            initial_value = self.trajectories[i].Get_y(0)
            chr_function = chrono.ChFunction_Const(float(initial_value))
            joint[1].joint.SetTorqueFunction(chr_function)

    def update_functions(self, time, robot_data, environment_data):
        for i, joint in enumerate(self.joints):
            current_value = self.trajectories[i].Get_y(time)
            chr_function = chrono.ChFunction_Const(float(current_value))
            joint[1].joint.SetTorqueFunction(chr_function)


class RobotControllerAngleTrajectoryChrono(RobotControllerChrono):

    def __init__(self, joint_vector, parameters: Dict[int, Any], trajectories=None):
        super().__init__(joint_vector, parameters, trajectories)

    def initialize_functions(self):
        if len(self.parameters) != len(self.joints):
            raise Exception("some joints are not parametrized")

        for i, joint in enumerate(self.joints):
            initial_value = self.trajectories[i].Get_y(0)
            chr_function = chrono.ChFunction_Const(float(initial_value))
            joint[1].joint.SetAngleFunction(chr_function)

    def update_functions(self, time, robot_data, environment_data):
        for i, joint in enumerate(self.joints):
            current_value = self.trajectories[i].Get_y(time)
            chr_function = chrono.ChFunction_Const(float(current_value))
            joint[1].joint.SetAngleFunction(chr_function)
