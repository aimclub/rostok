from copy import deepcopy
from tokenize import Double
import pychrono as chrono
import scipy.interpolate as interpolate
import numpy as np
from abc import ABC
import enum as Enum

from node_render import ChronoRevolveJoint

def define_inputs(joint,des_input,type):
    if type is ChronoRevolveJoint.InputType.Torque:
        joint.SetTorqueFunction(des_input)
    elif type is ChronoRevolveJoint.InputType.Velocity:
        joint.SetSpeedFunction(des_input)
    elif type is ChronoRevolveJoint.InputType.Position:
        joint.SetAngleFunction(des_input)
    elif type is ChronoRevolveJoint.InputType.Uncontrol:
        print("Uncontrollable joint")

def points_2_ChFunction(des_points, time_interval: tuple):
    num_points  = np.size(des_points)
    traj_time = np.linspace(time_interval[0],time_interval[1],num_points)

    spline_coefficients = interpolate.splrep(traj_time,des_points)
    out_func = ChTrajectory(spline_coefficients,time_interval)

    return out_func

class ChTrajectory(chrono.ChFunction):
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


class Control(chrono.ChFunction,ABC):
    def __init__(self,joint_block):
        self.__joint = joint_block

    def get_joint(self):
        return self.__joint.joint

    def get_type_input(self):
        return self.__joint.input_type

class RampControl(Control, chrono.ChFunction_Ramp):
    def __init__(self,joint_block, y_0 = 0, angular = 0.5):
        Control.__init__(self,joint_block)
        chrono.ChFunction_Ramp.__init__(y_0, angular)

class TrackingControl(Control):
    def __init__(self,joint_block,des_points, time_interval):
        Control.__init__(self,joint_block)
        self.time_interval = time_interval
        self.chr_trajectory = points_2_ChFunction(des_points,self.time_interval)
        define_inputs(self.get_joint(),self.chr_trajectory,self.get_type_input())
