from abc import abstractmethod
from math import sin
from typing import Dict, List

import pychrono.core as chrono
import keyboard
from rostok.block_builder_chrono.block_classes import (ChronoRevolveJoint,
                                                       JointInputTypeChrono)
from rostok.virtual_experiment.built_graph_chrono import BuiltGraphChrono


class RobotControllerChrono:
    """General controller. Any controller should be subclass of this class.
    
        Attributes:
            joints (List[Tuple[int, ChronoRevolveJoint]]): list of all joints in the mechanism
            parameters: vector of parameters for joints
            trajectories: trajectories for the joints
            functions: list of functions currently attached to joints
    """

    def __init__(self, built_graph: BuiltGraphChrono, parameters = {}):
        """Initialize class fields and call the initialize_functions() to set starting state"""
        self.built_graph = built_graph
        self.graph = built_graph.graph
        self.joint_map_ordered: Dict[int, ChronoRevolveJoint] = built_graph.joint_map_ordered
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
        self.active_j_to_fun_id = {}
        for idx, joint in self.joint_map_ordered.items():
            if self.chrono_joint_setters[joint.input_type] == 'Uncontrol':
                pass
            else:
                if 'initial_value' in self.parameters.keys():
                    chr_function = chrono.ChFunction_Const(float(self.parameters["initial_value"][i]))
                else:
                    chr_function = chrono.ChFunction_Const(0)
                joint_setter = getattr(joint.joint, self.chrono_joint_setters[joint.input_type])
                joint_setter(chr_function)
                self.functions.append(chr_function)
                self.active_j_to_fun_id[idx] = i
                i += 1

    @abstractmethod
    def update_functions(self, time, robot_data, environment_data):
        pass


class ConstController(RobotControllerChrono):

    def update_functions(self, time, robot_data, environment_data):
        pass

class SimpleKeyBoardController(RobotControllerChrono):
    def __init__(self, built_graph: BuiltGraphChrono, parameters):
        super().__init__(built_graph, parameters)
        
        self.forward_torque = self.parameters["forward"]
        self.reverse_torque = self.parameters["reverse"]
        
        self.forward_rotate = self.parameters["forward_rotate"]
        self.reverse_rotate = self.parameters["reverse_rotate"] 
  
    def update_functions(self, time, robot_data, environment_data):
        left_wheel = []
        right_wheel = []
        ordered_list_id = list(self.active_j_to_fun_id.keys())

        for number, joint_id in enumerate(self.joint_map_ordered):
            self.joint_map_ordered[joint_id].name
            side_indicator = str(self.joint_map_ordered[joint_id].name).split('_')[0]
            if side_indicator == "RM":
                right_wheel.append(joint_id)
            elif side_indicator == "LM":
                left_wheel.append(joint_id)
            else:
                pass
 
        
        if keyboard.is_pressed('a'):
            for func, i in zip(self.functions, ordered_list_id):
                if i in right_wheel:
                    func.Set_yconst(self.forward_rotate)
                if i in left_wheel:
                    func.Set_yconst(-self.reverse_rotate)
 
        elif keyboard.is_pressed('d'):
            for func, i in zip(self.functions, ordered_list_id):
                if i in left_wheel:
                    func.Set_yconst(self.forward_rotate)
                if i in right_wheel:
                    func.Set_yconst(-self.reverse_rotate)

        elif keyboard.is_pressed('w'):
            for  func in self.functions:
                func.Set_yconst(self.forward_torque)
        
        elif keyboard.is_pressed('s'):
            for  func in self.functions:
                func.Set_yconst(-self.reverse_torque)
        
        else:
            for func in self.functions:
                func.Set_yconst(0)





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
