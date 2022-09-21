from copy import deepcopy
import pychrono as chrono
import scipy.interpolate as interpolate
import numpy as np
from abc import ABC
from node_render import *

from node_render import ChronoRevolveJoint

def get_controllable_joints(blocks : list[Block]):
    #control_joints = filter(lambda x: isinstance(x,ChronoRevolveJoint), blocks)
    control_joints = list()
    for path in blocks:
        line = []
        for n in path:
            if isinstance(n,ChronoRevolveJoint):
                line.append(n)
        control_joints.append(line)
    return control_joints

# Transform 2-D array [time, points] to ChSpline subclass ChFunction
def time_points_2_ch_function(arr_time_point):
    arr_time_point = arr_time_point.T
    spline_coefficients = interpolate.splrep(arr_time_point[:,0],arr_time_point[:,1])
    time_interval = (arr_time_point[0,0],arr_time_point[-1,0])
    out_func = ChSpline(spline_coefficients,time_interval)

    return out_func

# Transform 1-D array on the time interval to ChSpline subclass ChFunction
def points_2_ch_function(des_points, time_interval: tuple):
    num_points  = np.size(des_points)
    trajectory_time = np.linspace(time_interval[0],time_interval[1],num_points)

    spline_coefficients = interpolate.splrep(trajectory_time,des_points)
    out_func = ChSpline(spline_coefficients,time_interval)

    return out_func

# Create custom function trajectory 
class ChCustomFunction(chrono.ChFunction):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        
    def Clone(self):
        return deepcopy(self)
    
    def Get_y(self, x):
        y = self.function(x, *self.args, **self.kwargs)
        return y

# Subclass ChFunction for set movement's trajectory
class ChSpline(chrono.ChFunction):
    def __init__(self, coefficients, time_interval):
        super().__init__()
        self.coefficients = coefficients
        self.time_interval = time_interval

    def Clone(self):
        return deepcopy(self)

    def Get_y(self, x):
        y = interpolate.splev(x, self.coefficients) 
        if self.time_interval[0] > x:
            y = interpolate.splev(self.time_interval[0], self.coefficients)
        if self.time_interval[1] < x:
            y = interpolate.splev(self.time_interval[1], self.coefficients)
        return float(y)

# Subclass ChFunction. PID-function calculate inputs for current time
class ChPIDposition(chrono.ChFunction_SetpointCallback):
    def __init__(self, joint, proportional_coefficient, differential_coefficient, integral_coefficients = 0.):
        super().__init__()
        self.K_P = proportional_coefficient
        self.K_D = differential_coefficient
        self.K_I = integral_coefficients
        self.err_i = 0
        self.joint = joint
        self.prev_time = 0.
        self.des_pos = 0.
        self.F = 0

    def Clone(self):
        return deepcopy(self)

    def set_des_point(self, des_position):
        self.des_pos = des_position


    def SetpointCallback(self, x):
        time = x
        if self.prev_time < time:
            mes_pos = self.joint.GetMotorRot()
            err =  self.des_pos.Get_y(time) - mes_pos
            
            mes_vel = self.joint.GetMotorRot_dt()
            d_err = self.des_pos.Get_y_dx(time) - mes_vel

            self.err_i += err*(time - self.prev_time)
            
            self.F = self.K_P*err + self.K_D*d_err + self.K_I*self.err_i
            self.prev_time = time
        return self.F

# Abstract class control
class Control(ABC):
    def __init__(self,joint_block):
        self.__joint = joint_block
        self.type_variants = {ChronoRevolveJoint.InputType.Torque: lambda x: self.get_joint().SetTorqueFunction(x),
                              ChronoRevolveJoint.InputType.Velocity: lambda x: self.get_joint().SetSpeedFunction(x),
                              ChronoRevolveJoint.InputType.Position: lambda x: self.get_joint().SetAngleFunction(x),
                              ChronoRevolveJoint.InputType.Uncontrol: None}

    def get_joint(self):
        return self.__joint.joint

    def get_type_input(self):
        return self.__joint.input_type

# Define type of input joint
    def set_input(self, inputs):
        try: 
            self.type_variants[self.get_type_input()](inputs)
        except TypeError:
           raise Exception(f"{self.get_joint()} is uncontrollable joint")

# Class ramp input on joint
class RampControl(Control):
    def __init__(self,in_joint_block, y_0 = 0., angular = -0.5):
        Control.__init__(self,in_joint_block)
        self.chr_function =  chrono.ChFunction_Ramp(y_0, angular)
        self.set_input(self.chr_function)

# Class tracking the trajectory at the position input
class TrackingControl(Control):
    def __init__(self,in_joint_block):
        Control.__init__(self,in_joint_block)
        self.time_interval = None
        self.chr_function = None
        
    def set_des_positions_interval(self, des_positions : np.array, time_interval: tuple):
        self.time_interval = time_interval
        self.chr_function = points_2_ch_function(des_positions, self.time_interval)
        self.set_input(self.chr_function)
        
    def set_des_positions(self, des_arr_time_to_pos : np.array):
        self.chr_function = time_points_2_ch_function(des_arr_time_to_pos)
        self.set_input(self.chr_function)
        
    def set_function_trajectory(self, function, *args, **kwargs):
        self.chr_function = ChCustomFunction(function, *args, **kwargs)
        self.set_input(self.chr_function)


# Class of PID tracking controller
class ChControllerPID(Control):
    def __init__(self, joint_block, proportional_coefficient, differential_coefficient, integral_coefficients = 0.):
        Control.__init__(self,joint_block)
        self.trajectory = None
        self.PID_ctrl = ChPIDposition(self.get_joint(), proportional_coefficient, differential_coefficient, integral_coefficients)

    def set_des_positions_interval(self, des_pos : np.array, time_interval: tuple):
        self.trajectory = points_2_ch_function(des_pos, time_interval)
        self.PID_ctrl.set_des_point(self.trajectory)
        self.set_input(self.PID_ctrl)

    def set_des_time_positions(self, des_arr_time_to_pos: np.array):
        self.trajectory = time_points_2_ch_function(des_arr_time_to_pos)
        self.PID_ctrl.set_des_point(self.trajectory)
        self.set_input(self.PID_ctrl)
        
    def set_function_trajectory(self, function, *args, **kwargs):
        self.trajectory = ChCustomFunction(function, *args, **kwargs)
        self.PID_ctrl.set_des_point(self.trajectory)
        self.set_input(self.PID_ctrl)
