from math import sin
from typing import Any, Dict, List, Tuple
from abc import abstractmethod
import pychrono.core as chrono

from rostok.block_builder_chrono.block_classes import (ChronoRevolveJoint, JointInputTypeChrono)
from rostok.virtual_experiment.sensors import Sensor
from scipy import interpolate

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
        self.set_function()
        self.initialize_functions()

    def set_function(self):
        self.chrono_joint_setters = {
            JointInputTypeChrono.TORQUE: 'SetTorqueFunction',
            JointInputTypeChrono.VELOCITY: 'SetSpeedFunction',
            JointInputTypeChrono.POSITION: 'SetAngleFunction',
            JointInputTypeChrono.UNCONTROL: 'Uncontrol'
        }
    @abstractmethod
    def initialize_functions(self):
        pass

    @abstractmethod
    def update_functions(self, time, robot_data, environment_data):
        pass


class ConstController(RobotControllerChrono):
    def initialize_functions(self):
        """Attach initial functions to the joints."""
        i = 0
        for idx, joint in self.joint_map_ordered.items():
            chr_function = chrono.ChFunction_Const(float(self.parameters["initial_value"][i]))
            joint_setter = getattr(joint.joint, self.chrono_joint_setters[joint.input_type])
            joint_setter(chr_function)
            self.functions.append(chr_function)
            i += 1

    def update_functions(self, time, robot_data, environment_data):
        pass

class PIDFunction(chrono.ChFunction):
    def __init__(self, K_p:float, K_d:float, K_i:float, reference: chrono.ChFunction):
        super().__init__()
        self.K_p = K_p
        self.K_d = K_d
        self.K_i = K_i
        self.reference = reference
        self.trajectory = []
        self.calculated_values = []
        self.calculated_times = [0]
        self.total_error = 0
        self.function = interpolate.interp1d([0],[0])

    def get_prev_time(self):
        return self.calculated_times[-1]

    def get_last_value(self):
        return self.calculated_values[-1]

    def update_function(self):
        self.function = interpolate.interp1d(self.calculated_times, self.calculated_values)

    def update(self, time, current_angle, current_angular_velocity):
        err = self.reference.Get_y(time) - current_angle
        d_err = self.reference.Get_y_dx(time) - current_angular_velocity
        self.total_error += err * (time - self.get_prev_time())
        result = self.K_p * err + self.K_d*d_err + self.K_i*self.total_error
        self.calculated_times.append(time)
        self.calculated_values.append(result)
        self.update_function()

    def Get_y(self, time):
        if time > self.get_prev_time():
            print("function called for time later than calculated values")
            return self.get_last_value()
        else:
            return float(self.function(time))

class PIDController(RobotControllerChrono):
    def initialize_functions(self):
        i = 0
        for idx, joint in self.joint_map_ordered.items():
            joint_parameters = self.parameters["PID_parameters"][i]
            chr_function = PIDFunction(joint_parameters[0], joint_parameters[1], joint_parameters[2], joint_parameters[3])
            chr_function.calculated_values.append(joint_parameters[4])
            joint_setter = getattr(joint.joint, self.chrono_joint_setters[joint.input_type])
            joint_setter(chr_function)
            self.functions.append(chr_function)
            i += 1

    def update_functions(self, time, robot_data:Sensor, environment_data):
        i = 0
        for idx, _ in self.joint_map_ordered.items():
            current_angle = robot_data.get_active_joint_trajectory_point()[idx]
            current_angular_speed = robot_data.get_active_joint_speed()[idx]
            func = self.functions[i]
            func.update(time, current_angle, current_angular_speed)


class SinControllerChrono(RobotControllerChrono):
    """Controller that sets sinusoidal torques using constant update at each step."""
    def initialize_functions(self):
        """Attach initial functions to the joints."""
        i = 0
        for idx, joint in self.joint_map_ordered.items():
            chr_function = chrono.ChFunction_Const(float(self.parameters["initial_value"][i]))
            joint_setter = getattr(joint.joint, self.chrono_joint_setters[joint.input_type])
            joint_setter(chr_function)
            self.functions.append(chr_function)
            i += 1
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


# class TorqueTrajectoryControllerChrono(RobotControllerChrono):
#     def __init__(self, joint_map_ordered, parameters: Dict[int, Any], trajectories):
#         super().__init__(joint_map_ordered, parameters, trajectories)

#     def update_functions(self, time, robot_data: Sensor, environment_data):
#         for i, trajectory in enumerate(self.trajectories):
#             if time > trajectory[0][1]
