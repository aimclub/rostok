from math import sin
from typing import Any, Dict, List, Tuple

import pychrono.core as chrono

from rostok.block_builder_chrono.block_classes import (ChronoRevolveJoint,
                                                       JointInputTypeChrono)
from rostok.virtual_experiment.sensors import Sensor


class RobotControllerChrono:

    def __init__(self, joint_vector, parameters: Dict[int, Any], trajectories=None):
        self.joints: List[Tuple[int, ChronoRevolveJoint]] = joint_vector
        self.parameters = parameters
        self.trajectories = trajectories
        self.initialize_functions()

    def get_joint_by_id(self, idx: int):
        for joint in self.joints:
            if joint[0] == idx:
                return joint[1]
        return None

    def initialize_functions(self):
        if len(self.parameters) != len(self.joints):
            raise Exception("some joints are not parametrized")

        for i, joint in enumerate(self.joints):
            chr_function = chrono.ChFunction_Const(float(self.parameters[i]))
            joint[1].joint.SetTorqueFunction(chr_function)

    def update_functions(self, time, robot_data, environment_data):
        pass


class SinControllerChronoFn(RobotControllerChrono):

    def __init__(self, joint_vector, parameters: Dict[int, Any], trajectories=None):
        super().__init__(joint_vector, parameters, trajectories)

    def initialize_functions(self):
        parameters = self.parameters
        if len(parameters) != len(self.joints):
            raise Exception("some joints are not parametrized")
        for i, joint in enumerate(self.joints):
            #joint[1].joint.SetTorqueFunction(chrono.ChFunction_Const(0.0))
            chr_function = chrono.ChFunction_Sine(0.0, parameters[i][1] / 6.28, parameters[i][0])
            joint[1].joint.SetTorqueFunction(chr_function)

    def update_functions(self, time, robot_data, environment_data):
        # for i, joint in enumerate(self.joints):
        #     current_const = parameters[i][0]*sin(parameters[i][1]*time)
        #     chr_function = chrono.ChFunction_Const(current_const)

        #     joint[1].joint.SetTorqueFunction(chr_function)
        pass


class SinControllerChrono(RobotControllerChrono):

    def __init__(self, joint_vector, parameters: Dict[int, Any], trajectories=None):
        super().__init__(joint_vector, parameters, trajectories)

    def initialize_functions(self):
        if len(self.parameters) != len(self.joints):
            raise Exception("some joints are not parametrized")
        for i, joint in enumerate(self.joints):
            joint[1].joint.SetTorqueFunction(chrono.ChFunction_Const(0.0))

    def update_functions(self, time, robot_data, environment_data):
        for i, joint in enumerate(self.joints):
            current_const = self.parameters[i][0] * sin(self.parameters[i][1] * time)
            chr_function = chrono.ChFunction_Const(current_const)

            joint[1].joint.SetTorqueFunction(chr_function)


class ConstReverseControllerChrono(RobotControllerChrono):

    def __init__(self, joint_vector, parameters: Dict[int, Any], trajectories=None):
        super().__init__(joint_vector, parameters, trajectories=None)
        self.change_prev_step = False

    def initialize_functions(self):
        if len(self.parameters) != len(self.joints):
            raise Exception("some joints are not parametrized")
        for i, joint in enumerate(self.joints):
            chr_function = chrono.ChFunction_Const(float(self.parameters[i]))
            joint[1].joint.SetTorqueFunction(chr_function)

    def update_functions(self, time, robot_data: Sensor, environment_data):
        for item in robot_data.joint_body_map.items():
            if not robot_data.amount_contact_forces(
                    item[1][1]) is None and not self.change_prev_step:
                joint: ChronoRevolveJoint = self.get_joint_by_id(item[0])
                current_const = joint.joint.GetTorqueFunction().Get_y(0)
                joint.joint.SetTorqueFunction(chrono.ChFunction_Const(-current_const))
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
