from math import sin
from typing import Any, Dict, List, Tuple
from abc import abstractmethod
import pychrono.core as chrono

from rostok.block_builder_chrono.block_classes import (ChronoRevolveJoint,
                                                       JointInputTypeChrono)
from rostok.virtual_experiment.sensors import Sensor


class RobotControllerChrono:
    """General controller. Any controller should be subclass of this class.
    
        Attributes:
            joints (List[Tuple[int, ChronoRevolveJoint]]): list of all joints in the mechanism
            parameters: vector of parameters for joints
            trajectories: trajectories for the joints
            functions: list of functions currently attached to joints
    """
    def __init__(self, joint_map, parameters: Dict[str, Any], trajectories=None):
        """Initialize class fields and call the initialize_functions() to set starting state"""
        self.joint_map: Dict[int, ChronoRevolveJoint] = joint_map
        self.parameters = parameters
        self.trajectories = trajectories
        self.functions: List[chrono.ChFunction_Const] = []
        self.chrono_joint_setters:Dict[JointInputTypeChrono, str] = {}
        self.do_nothing = lambda x : None
        self.set_function()
        self.initialize_functions()

    def set_function(self):
        self.chrono_joint_setters = {
            JointInputTypeChrono.TORQUE : 'SetTorqueFunction',
            JointInputTypeChrono.VELOCITY : 'SetSpeedFunction',
            JointInputTypeChrono.POSITION: 'SetAngleFunction',
            JointInputTypeChrono.UNCONTROL: 'DoNothing'
        }

    def initialize_functions(self):
        """Attach initial functions to the joints."""
        for i, joint in enumerate(self.joint_map.items()):
            chr_function = chrono.ChFunction_Const(float(self.parameters["initial_value"][i]))
            getattr(joint[1].joint, self.chrono_joint_setters[joint[1].input_type], self.do_nothing)(chr_function)
            self.functions.append(chr_function)

    @abstractmethod
    def update_functions(self, time, robot_data, environment_data):
        pass

class ConstController(RobotControllerChrono):

    def update_functions(self, time, robot_data, environment_data):
        pass

class SinControllerChrono(RobotControllerChrono):
    """Controller that sets sinusoidal torques using constant update at each step."""

    def update_functions(self, time, robot_data, environment_data):
        for i, _ in enumerate(self.joint_map):
            current_const = self.parameters['sin_parameters'][i][0] * sin(self.parameters['sin_parameters'][i][1] * time)
            self.functions[i].Set_yconst(current_const)


class ConstReverseControllerChrono(RobotControllerChrono):

    def __init__(self, joint_map, parameters: Dict[int, Any], trajectories=None):
        super().__init__(joint_map, parameters, trajectories=None)
        self.change_prev_step = False

    def update_functions(self, time, robot_data: Sensor, environment_data):
        i = 0
        for item in robot_data.joint_body_map.items():
            if not robot_data.amount_contact_forces(
                    item[1][1]) is None and not self.change_prev_step:
                joint: ChronoRevolveJoint = self.joint_map(item[0])
                current_const = - joint.joint.GetTorqueFunction().Get_y(0)
                self.functions[i].Set_yconst(current_const)
                self.change_prev_step = True
                return

        self.change_prev_step = False

# class TorqueTrajectoryControllerChrono(RobotControllerChrono):
#     def __init__(self, joint_map, parameters: Dict[int, Any], trajectories):
#         super().__init__(joint_map, parameters, trajectories)
        
#     def update_functions(self, time, robot_data: Sensor, environment_data):
#         for i, trajectory in enumerate(self.trajectories):
#             if time > trajectory[0][1]
