from math import sin
from typing import Any, Dict, List, Tuple
from abc import abstractmethod
import pychrono.core as chrono

from rostok.block_builder_chrono_alt.block_classes import (ChronoRevolveJoint, JointInputTypeChrono)
from rostok.virtual_experiment.sensors import Sensor


class RobotControllerChrono:
    """General controller. Any controller should be subclass of this class.
    
        Attributes:
            joints (List[Tuple[int, ChronoRevolveJoint]]): list of all joints in the mechanism
            parameters: vector of parameters for joints
            trajectories: trajectories for the joints
            functions: list of functions currently attached to joints
    """

    def __init__(self, joint_map_ordered, parameters: Dict[str, Any]):
        """Initialize class fields and call the initialize_functions() to set starting state"""
        self.joint_map_ordered: Dict[int, ChronoRevolveJoint] = joint_map_ordered
        self.parameters = parameters
        self.functions: List[chrono.ChFunction_Const] = []
        self.chrono_joint_setters: Dict[JointInputTypeChrono, str] = {}
        self.do_nothing = lambda x: None
        self.set_function()
        self.initialize_functions()

    def set_function(self):
        self.chrono_joint_setters = {
            JointInputTypeChrono.TORQUE: 'SetTorqueFunction',
            JointInputTypeChrono.VELOCITY: 'SetSpeedFunction',
            JointInputTypeChrono.POSITION: 'SetAngleFunction',
            JointInputTypeChrono.UNCONTROL: 'Uncontrol'
        }

    def initialize_functions(self):
        """Attach initial functions to the joints."""
        i = 0
        for idx, joint in self.joint_map_ordered.items():
            if self.chrono_joint_setters[joint.input_type] == 'Uncontrol':
                pass
            else:
                chr_function = chrono.ChFunction_Const(float(self.parameters["initial_value"][i]))
                joint_setter = getattr(joint.joint, self.chrono_joint_setters[joint.input_type])
                joint_setter(chr_function)
                self.functions.append(chr_function)
                i += 1

    @abstractmethod
    def update_functions(self, time, robot_data, environment_data):
        pass


class ConstController(RobotControllerChrono):

    def update_functions(self, time, robot_data, environment_data):
        pass


class SinControllerChrono(RobotControllerChrono):
    """Controller that sets sinusoidal torques using constant update at each step."""

    def update_functions(self, time, robot_data, environment_data):
        for i, func in enumerate(self.functions):
            current_const = self.parameters['sin_parameters'][i][0] * sin(
                self.parameters['sin_parameters'][i][1] * time)
            func.Set_yconst(current_const)


class LinearSinControllerChrono(RobotControllerChrono):
    """Controller that sets sinusoidal torques using constant update at each step."""

    def update_functions(self, time, robot_data, environment_data):
        for i, func in enumerate(self.functions):
            current_const = self.parameters['sin_parameters'][i][2] * time * self.parameters[
                'sin_parameters'][i][0] * sin(self.parameters['sin_parameters'][i][1] * time)
            func.Set_yconst(current_const)

import numpy as np
class PIDController(RobotControllerChrono):
    def __init__(self):
        super().__init__()
        self.total_errors = list(np.zeros(len(self.joint_map_ordered)))
        self.references = []
    def initialize_functions(self):
        i = 0
        for idx, joint in self.joint_map_ordered.items():
            if self.chrono_joint_setters[joint.input_type] == 'Uncontrol':
                pass
            else:
                self.references[i] = self.parameters["PID_parameters"][i][3]
                starting_error = 0 - self.references[i].Get_y(0)
                starting_dt_error = 0 - self.references[i].Get_y_dx(0)
                self.total_errors[i] = starting_error
                initial_value = self.parameters["PID_parameters"][i][0] * starting_error + self.parameters["PID_parameters"][i][1]*starting_dt_error+self.parameters["PID_parameters"][i][2]*self.total_errors[i]
                chr_function = chrono.ChFunction_Const(initial_value)
                joint_setter = getattr(joint.joint, self.chrono_joint_setters[joint.input_type])
                joint_setter(chr_function)
                self.functions.append(chr_function)
                i += 1

    def update_functions(self, time, robot_data:Sensor, environment_data):
        for i, func in enumerate(self.functions):
            current_error = 
# class TorqueTrajectoryControllerChrono(RobotControllerChrono):
#     def __init__(self, joint_map_ordered, parameters: Dict[int, Any], trajectories):
#         super().__init__(joint_map_ordered, parameters, trajectories)

#     def update_functions(self, time, robot_data: Sensor, environment_data):
#         for i, trajectory in enumerate(self.trajectories):
#             if time > trajectory[0][1]
