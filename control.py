from copy import deepcopy
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

'''
class ChPID(chrono.ChFunction_SetpointCallback):
    def __init__(self, joint, coeff_p, coeff_d, coeff_i = 0.):
        super().__init__()
        self.K_P = coeff_p
        self.K_D = coeff_d
        self.K_I = coeff_i
        self.err_i = 0
        self.joint = joint
        self.prev_time = 0.
        self.des_pos = 0.
        self.des_vel = 0.
        self.des_acc = 0.
        self.F = 0

    def Clone(self):
        return deepcopy(self)

    def set_des_point(self, des_position = 0., des_velocity = 0., des_acceleration = 0.):
        self.des_pos = des_position
        self.des_vel = des_velocity
        self.des_acc = des_acceleration


    def SetpointCallback(self, x):
        time = x
        if (self.prev_time < time):
            mes_pos = self.joint.GetMotorRot()
            err =  self.des_pos - mes_pos
            
            mes_vel = self.joint.GetMotorRot_dt()
            d_err = self.des_vel - mes_vel

            self.err_i += err*(time - self.prev_time)
            
            self.F = self.K_P*err + self.K_D*d_err + self.K_I*self.err_i
            self.prev_time = time
        return self.F

'''
class ChPID(chrono.ChFunction_SetpointCallback):
    def __init__(self, joint, coeff_p, coeff_d, coeff_i = 0.):
        super().__init__()
        self.K_P = coeff_p
        self.K_D = coeff_d
        self.K_I = coeff_i
        self.err_i = 0
        self.joint = joint
        self.prev_time = 0.
        self.des_pos = 0.
        self.des_vel = 0.
        self.des_acc = 0.
        self.F = 0

    def Clone(self):
        return deepcopy(self)

    def set_des_point(self, des_position):
        self.des_pos = des_position


    def SetpointCallback(self, x):
        time = x
        if (self.prev_time < time):
            mes_pos = self.joint.GetMotorRot()
            err =  self.des_pos.Get_y(time) - mes_pos
            
            mes_vel = self.joint.GetMotorRot_dt()
            d_err = self.des_pos.Get_y_dx(time) - mes_vel

            self.err_i += err*(time - self.prev_time)
            
            self.F = self.K_P*err + self.K_D*d_err + self.K_I*self.err_i
            self.prev_time = time
        return self.F

class Control(ABC):
    def __init__(self,joint_block):
        self.__joint = joint_block

    def get_joint(self):
        return self.__joint.joint

    def get_type_input(self):
        return self.__joint.input_type

    def set_input(self, inputs):
        type_input = self.get_type_input()
        if type_input is ChronoRevolveJoint.InputType.Torque:
            self.get_joint().SetTorqueFunction(inputs)
        elif type_input is ChronoRevolveJoint.InputType.Velocity:
            self.get_joint().SetSpeedFunction(inputs)
        elif type_input is ChronoRevolveJoint.InputType.Position:
            self.get_joint().SetAngleFunction(inputs)
        elif type_input is ChronoRevolveJoint.InputType.Uncontrol:
            print("Uncontrollable joint")

class RampControl(Control):
    def __init__(self,in_joint_block, y_0 = 0., angular = -0.5):
        Control.__init__(self,in_joint_block)
        self.chr_trajectory =  chrono.ChFunction_Ramp(y_0, angular)
        self.set_input(self.chr_trajectory)

class TrackingControl(Control):
    def __init__(self,in_joint_block,des_points, time_interval):
        Control.__init__(self,in_joint_block)
        self.time_interval = time_interval
        self.chr_trajectory = points_2_ChFunction(des_points,self.time_interval)
        self.set_input(self.chr_trajectory)
'''
class ChControllerPID(Control):
    def __init__(self, joint_block, des_position, coeff_p, coeff_d, coeff_i = 0.):
        Control.__init__(self,joint_block)
        self.PID_ctrl = ChPID(self.get_joint(),coeff_p, coeff_d, coeff_i)
        self.PID_ctrl.set_des_point(des_position)
        self.set_input(self.PID_ctrl)
'''
class ChControllerPID(Control):
    def __init__(self, joint_block, des_position, coeff_p, coeff_d, coeff_i = 0.):
        Control.__init__(self,joint_block)
        self.PID_ctrl = ChPID(self.get_joint(),coeff_p, coeff_d, coeff_i)
        self.traj = points_2_ChFunction(des_position,(0.5, 5))
        self.PID_ctrl.set_des_point(self.traj)
        self.set_input(self.PID_ctrl)