from rostok.block_builder.node_render import ChronoRevolveJoint, Block

from copy import deepcopy
import pychrono as chrono
import scipy.interpolate as interpolate
import numpy as np
from abc import ABC


def get_controllable_joints(blocks: list[Block]):
    """Create 2D-list joints from list of blocks. First index is the number
        partition graph, second index is the number of joint

    Args:
        blocks (list[Block]): List of blocks from graph
    Returns:
        list[list]: List of joints
    """
    control_joints = []
    for path in blocks:
        line = []
        for n in path:
            if isinstance(n, ChronoRevolveJoint):
                control_joints.append(n)
    #control_joint = filter(lambda x: isinstance(x,ChronoRevolveJoint), blocks)
    return control_joints



def time_points_2_ch_function(arr_time_point):
    """Transform 2-D array [time, points] to ChSpline subclass ChFunction

    Args:
        arr_time_point (list[list]): 2D-list of values(times)

    Returns:
        ChSpline: Chrono function of spline points
    """
    arr_time_point = arr_time_point.T
    spline_coefficients = interpolate.splrep(
        arr_time_point[:, 0], arr_time_point[:, 1])
    time_interval = (arr_time_point[0, 0], arr_time_point[-1, 0])
    out_func = ChSpline(spline_coefficients, time_interval)

    return out_func


def points_2_ch_function(des_points, time_interval: tuple):
    """Transform 1-D array on the time interval to ChSpline subclass ChFunction

    Args:
        des_points (list): List of values which define spline
        time_interval (tuple): Interval between time start and finish points 

    Returns:
        ChSpline: Chrono function of spline points
    """
    num_points = np.size(des_points)
    trajectory_time = np.linspace(
        time_interval[0], time_interval[1], num_points)

    spline_coefficients = interpolate.splrep(trajectory_time, des_points)
    out_func = ChSpline(spline_coefficients, time_interval)

    return out_func


class ChCustomFunction(chrono.ChFunction):

    def __init__(self, function, *args, **kwargs):
        """Create custom chrono function
        Args:
            function (link to function): Link to function which convert to chrono function
        """
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def Clone(self):
        return deepcopy(self)

    def Get_y(self, x):
        y = self.function(x, *self.args, **self.kwargs)
        return y


class ChSpline(chrono.ChFunction):
    def __init__(self, coefficients, time_interval):
        """Class of spline function on chrono

        Args:
            coefficients (list): List of coefficients cubic spline
            time_interval (tuple): Time interval of spline
        """
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


class ChPID(chrono.ChFunction_SetpointCallback):
    def __init__(self, joint, proportional_coefficient, differential_coefficient, integral_coefficients=0.):
        """Subclass ChFunction. PID-function calculate inputs for current time

        Args:
            joint (ChMotor): Chrono Motor of joint
            proportional_coefficient (float): proportional coefficient of PID
            differential_coefficient (float): differential coefficient of PID
            integral_coefficients (float): integral coefficient of PID. Defaults to 0..
        """
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
        """Setter the value against which the error is calculate

        Args:
            des_position (float): Value of desired point to controller value
        """
        self.des_pos = des_position

    def SetpointCallback(self, x):
        time = x
        if self.prev_time < time:
            mes_pos = self.joint.GetMotorRot()
            err = self.des_pos.Get_y(time) - mes_pos

            mes_vel = self.joint.GetMotorRot_dt()
            d_err = self.des_pos.Get_y_dx(time) - mes_vel

            self.err_i += err*(time - self.prev_time)

            self.F = self.K_P*err + self.K_D*d_err + self.K_I*self.err_i
            self.prev_time = time
        return self.F

class ChronoControl(ABC):
    def __init__(self, joint_block):
        """Abstract class of control for chrono joint (ChMotor)

        Args:
            joint_block (ChMotor): Object to create the controller 
        """
        self.__joint = joint_block
        self.type_variants = {ChronoRevolveJoint.InputType.TORQUE: lambda x: self.get_joint().SetTorqueFunction(x),
                              ChronoRevolveJoint.InputType.VELOCITY: lambda x: self.get_joint().SetSpeedFunction(x),
                              ChronoRevolveJoint.InputType.POSITION: lambda x: self.get_joint().SetAngleFunction(x),
                              ChronoRevolveJoint.InputType.UNCONTROL: None}

    def get_joint(self):
        return self.__joint.joint

    def get_type_input(self):
        return self.__joint.input_type

    def set_input(self, inputs):
        """Define type of input joint

        Args:
            inputs (ChFunction): Chrono function of input controller

        Raises:
            Exception: If joints is uncontrollable
        """
        try:
            self.type_variants[self.get_type_input()](inputs)
        except TypeError:
            raise Exception(f"{self.get_joint()} is uncontrollable joint")


class RampControl(ChronoControl):
    def __init__(self, in_joint_block, y_0=0., angular=-0.5):
        """Class ramp input on joint

        Args:
            in_joint_block (ChMotor): Object to create the controller
            y_0 (float): Initial values ramp function. Defaults to 0..
            angular (float): Angle of ramp function. Defaults to -0.5.
        """
        ChronoControl.__init__(self, in_joint_block)
        self.chr_function = chrono.ChFunction_Ramp(y_0, angular)
        self.set_input(self.chr_function)

class TrackingControl(ChronoControl):
    def __init__(self, in_joint_block):
        """Class tracking the trajectory at the position input

        Args:
            in_joint_block (ChMotor)): Object to create controller
        """
        ChronoControl.__init__(self, in_joint_block)
        self.time_interval = None
        self.chr_function = None

    def set_des_positions_interval(self, des_positions: np.array, time_interval: tuple):
        """Setter the desired trajectory on time interval

        Args:
            des_positions (np.array): Desired trajectory on the points
            time_interval (tuple): (start, stop) time interval of trajectory
        """
        self.time_interval = time_interval
        self.chr_function = points_2_ch_function(
            des_positions, self.time_interval)
        self.set_input(self.chr_function)

    def set_des_positions(self, des_arr_time_to_pos: np.array):
        """Setter the desired trajectory how 2D array [[time, position],...]

        Args:
            des_arr_time_to_pos (np.array): 2D array time value of the desired trajectory
        """
        self.chr_function = time_points_2_ch_function(des_arr_time_to_pos)
        self.set_input(self.chr_function)

    def set_function_trajectory(self, function, *args, **kwargs):
        """Setter the desired trajectory how python function

        Args:
            function (def): Link to the function which generate values the desired trajectory each time
        """
        self.chr_function = ChCustomFunction(function, *args, **kwargs)
        self.set_input(self.chr_function)



class ChControllerPID(ChronoControl):
    def __init__(self, joint_block, proportional_coefficient, differential_coefficient, integral_coefficients=0.):
        """Class of PID tracking controller

        Args:
            joint (ChMotor): Chrono Motor of joint
            proportional_coefficient (float): proportional coefficient of PID
            differential_coefficient (float): differential coefficient of PID
            integral_coefficients (float): integral coefficient of PID. Defaults to 0..
        """
        ChronoControl.__init__(self, joint_block)
        self.trajectory = None
        self.PID_ctrl = ChPID(self.get_joint(
        ), proportional_coefficient, differential_coefficient, integral_coefficients)

    def set_des_positions_interval(self, des_pos: np.array, time_interval: tuple):
        """Setter the desired trajectory on time interval

        Args:
            des_positions (np.array): Desired trajectory on the points
            time_interval (tuple): (start, stop) time interval of trajectory
        """
        self.trajectory = points_2_ch_function(des_pos, time_interval)
        self.PID_ctrl.set_des_point(self.trajectory)
        self.set_input(self.PID_ctrl)

    def set_des_time_positions(self, des_arr_time_to_pos: np.array):
        """Setter the desired trajectory how 2D array [[time, position],...]

        Args:
            des_arr_time_to_pos (np.array): 2D array time value of the desired trajectory
        """
        self.trajectory = time_points_2_ch_function(des_arr_time_to_pos)
        self.PID_ctrl.set_des_point(self.trajectory)
        self.set_input(self.PID_ctrl)

    def set_function_trajectory(self, function, *args, **kwargs):
        """Setter the desired trajectory how python function

        Args:
            function (def): Link to the function which generate values the desired trajectory each time
        """
        self.trajectory = ChCustomFunction(function, *args, **kwargs)
        self.PID_ctrl.set_des_point(self.trajectory)
        self.set_input(self.PID_ctrl)

class ConstControl(ChronoControl):
    def __init__(self, in_joint_block, T=0.):
        ChronoControl.__init__(self, in_joint_block)
        self.chr_function = chrono.ChFunction_Const(T)
        self.set_input(self.chr_function)
