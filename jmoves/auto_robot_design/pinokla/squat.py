from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Callable
from auto_robot_design.pinokla.closed_loop_jacobian import ConstraintFrameJacobian
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
import pinocchio as pin
import numpy as np
import meshcat
from pinocchio.visualize import MeshcatVisualizer

from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopInverseKinematicsProximal, closedLoopProximalMount
import numpy as np


from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
from auto_robot_design.pinokla.robot_utils import add_3d_constrain_current_q


def quartic_func_free_acc(q0, qf, T, qd0=0, qdf=0):
    """
    Quartic scalar polynomial as a function.
    Final acceleration is unconstrained, start acceleration is zero.

    :param q0: initial value
    :type q0: float
    :param qf: final value
    :type qf: float
    :param T: trajectory time
    :type T: float
    :param qd0: initial velocity, defaults to 0
    :type q0: float, optional
    :param qdf: final velocity, defaults to 0
    :type q0: float, optional
    :return: polynomial function :math:`f: t \mapsto (q(t), \dot{q}(t), \ddot{q}(t))`
    :rtype: callable

    Returns a function which computes the specific quartic polynomial, and its
    derivatives, as described by the parameters.


    """

    # solve for the polynomial coefficients using least squares
    # fmt: off
    X = [
        [0.0,         0.0,        0.0,     0.0,  1.0],
        [T**4,        T**3,       T**2,    T,    1.0],
        [0.0,         0.0,        0.0,     1.0,  0.0],
        [4.0 * T**3,  3.0 * T**2, 2.0 * T, 1.0,  0.0],
        [0.0,         0.0,        2.0,     0.0,  0.0],

    ]
    # fmt: on
    coeffs, resid, rank, s = np.linalg.lstsq(X,
                                             np.r_[q0, qf, qd0, qdf, 0],
                                             rcond=None)

    # coefficients of derivatives
    coeffs_d = coeffs[0:4] * np.arange(4, 0, -1)
    coeffs_dd = coeffs_d[0:3] * np.arange(3, 0, -1)

    return lambda x: (
        np.polyval(coeffs, x),
        np.polyval(coeffs_d, x),
        np.polyval(coeffs_dd, x),
    )


def calculate_final_v(desired_hop_high: float):
    g = 9.81
    return np.sqrt(desired_hop_high * 2 * g)


class HopDirection(IntFlag):
    Z = auto()
    Y = auto()
    X = auto()


@dataclass
class SquatHopParameters:
    """    
    hop_flight_hight -- describes how far the robot flies after liftoff
    total_time: float -- total time of simulation
    squatting_down_hight -- coordinate of robot base in start
    of squatting respect of base in nominal pose  
    squatting_up_hight -- coordinate of robot base in end
    of squatting respect of base in nominal pose
    hop_direction -- now implemented only z 
    end_effector_name 
    ground_link_name 

    """
    hop_flight_hight: float = 0.2
    total_time: float = 0.7
    squatting_down_hight: float = -0.2
    squatting_up_hight: float = 0.2
    hop_direction: HopDirection = HopDirection.Z
    end_effector_name: str = "EE"
    ground_link_name: str = "G"


class SimulateSquatHop:

    def __init__(self, squat_hop_parameters: SquatHopParameters) -> None:
        self.squat_hop_parameters = squat_hop_parameters

    def set_robot(self,
                  robo_urdf: str,
                  joint_description: dict,
                  loop_description: dict,
                  actuator_context=None):
        """Initialized two types of model. 
        1) simple model with fixed base 
        2) model with 3d constrained end-effector plus base on prismatic joint 
        Different model needed for correct solve inverse kinematic. 
        Coordinates associated with base joint located in [0] position in q vector.

        Args:
            robo_urdf (str): _description_
            joint_description (dict): _description_
            loop_description (dict): _description_
            actuator_context (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplemented: Only z axis implemented
        """
        self.fixed_base_robo = build_model_with_extensions(
            robo_urdf,
            joint_description=joint_description,
            loop_description=loop_description,
            actuator_context=None,
            fixed=True)

        self.trans_base_robo = build_model_with_extensions(
            robo_urdf,
            joint_description=joint_description,
            loop_description=loop_description,
            actuator_context=None,
            fixed=False,
            root_joint_type=pin.JointModelTranslation(),
            is_act_root_joint=False)

        q_nominal = self.calc_nominal_q()
        qushka = np.concatenate([np.zeros(2), q_nominal])
        # qushka = q_nominal
        self.trans_base_robo = add_3d_constrain_current_q(
            self.trans_base_robo, self.squat_hop_parameters.end_effector_name,
            qushka)

        if self.squat_hop_parameters.hop_direction == HopDirection.Z:
            root_joint_type = pin.JointModelPZ()
        else:
            raise NotImplemented

        self.hop_robo = build_model_with_extensions(
            robo_urdf,
            joint_description=joint_description,
            loop_description=loop_description,
            actuator_context=actuator_context,
            fixed=False,
            root_joint_type=root_joint_type,
            is_act_root_joint=False)

        q_nominal = self.calc_nominal_q()
        self.hop_robo = add_3d_constrain_current_q(
            self.hop_robo, self.squat_hop_parameters.end_effector_name,
            q_nominal)

        self.robo_urdf = robo_urdf
        self.joint_description = joint_description
        self.loop_description = loop_description
        self.actuator_context = actuator_context

    def calc_nominal_q(self):
        """Calculate q vector for nominal pose.
        Vector is applicable for self.hop_robo.

        Returns:
            _type_: _description_
        """
        if len(self.fixed_base_robo.constraint_models) == 0:
        # Condition for check open kinematics
            q0 = np.array([np.deg2rad(0), np.deg2rad(0)])
        else:  
            q0 = closedLoopProximalMount(self.fixed_base_robo.model,
                                        self.fixed_base_robo.data,
                                        self.fixed_base_robo.constraint_models,
                                        self.fixed_base_robo.constraint_data)
        q0_plus_base_pos = self.add_base_pos(0, q0)
        return q0_plus_base_pos

    def add_base_pos(self, q_base: float, q_leg: np.ndarray) -> np.ndarray:
        """Extend a vector associated with leg by base_q.

        Args:
            q_base (float): associated with base
            q_leg (np.ndarray): associated with leg

        Returns:
            np.ndarray: q vector
        """
        return np.concatenate([np.array([q_base]), q_leg])

    def calc_start_squat_q(self) -> np.ndarray:
        """Calculate q is applicable for self.hop_robo. 
        Set robot in squat position. For solve inverse kinematic used
        self.fixed_base_robo.

        Raises:
            NotImplemented: _description_

        Returns:
              np.ndarray: q vector
        """
        nominal_q_hop_robot = self.calc_nominal_q()
        nominal_q = nominal_q_hop_robot[1:]

        pin.framesForwardKinematics(self.fixed_base_robo.model,
                                    self.fixed_base_robo.data, nominal_q)
        ee_name = self.squat_hop_parameters.end_effector_name
        ee_id = self.fixed_base_robo.model.getFrameId(ee_name)
        if self.squat_hop_parameters.hop_direction == HopDirection.Z:
            id_in_vector = 2
        else:
            raise NotImplemented

        default_hight = self.fixed_base_robo.data.oMf[ee_id].translation
        default_hight[id_in_vector] = default_hight[id_in_vector] - \
            self.squat_hop_parameters.squatting_down_hight

        needed_q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
            self.fixed_base_robo.model,
            self.fixed_base_robo.data,
            self.fixed_base_robo.constraint_models,
            self.fixed_base_robo.constraint_data,
            default_hight,
            ee_id,
            onlytranslation=True,
        )
        needed_q = self.add_base_pos(
            self.squat_hop_parameters.squatting_down_hight, needed_q)
        return needed_q, is_reach

    def setup_dynamic(self):
        """Initializes the dynamics calculator, also set time_step. 
        """
        accuracy = 1e-8
        mu_sim = 1e-8
        max_it = 10000
        DT = 10e-4
        self.dynamic_settings = pin.ProximalSettings(accuracy, mu_sim, max_it)
        self.time_step = DT

    def simulate(self,
                 robo_urdf: str,
                 joint_description: dict,
                 loop_description: dict,
                 control_coefficient: float = 0.8,
                 actuator_context=None,
                 is_vis=False):
        """Simulate squat and hop process. Uses method
        self.setup_dynamic  
        self.set_robot
        Args:
            robo_urdf (str): _description_
            joint_description (dict): _description_
            loop_description (dict): _description_
            actuator_context (_type_, optional): _description_. Defaults to None.

        Raises:
            Exception: _description_
         Returns:
              np.ndarray, np.ndarray, np.ndarray: position, velocity, acceleration 
        """
        self.setup_dynamic()
        self.set_robot(robo_urdf, joint_description, loop_description,
                       actuator_context)
        start_squat_q, is_reach = self.calc_start_squat_q()
        if not is_reach:
            raise Exception("Start squat position is not reached")
        traj_fun = self.create_traj_equation()

        pin.computeGeneralizedGravity(self.hop_robo.model, self.hop_robo.data,
                                      start_squat_q)
        grav_force = self.hop_robo.data.g[0]
        total_mass = pin.computeTotalMass(self.hop_robo.model)
        simulate_steps = int(self.squat_hop_parameters.total_time /
                             self.time_step)

        pin.initConstraintDynamics(self.hop_robo.model, self.hop_robo.data,
                                   self.hop_robo.constraint_models)
        vq = np.zeros(self.hop_robo.model.nv)
        q = start_squat_q
        tau_q = np.zeros(self.hop_robo.model.nv)

        q_act = np.zeros((simulate_steps, self.hop_robo.model.nv))
        vq_act = np.zeros((simulate_steps, self.hop_robo.model.nv))
        acc_act = np.zeros((simulate_steps, self.hop_robo.model.nv))
        tau_act = np.zeros((simulate_steps, 2))

        if is_vis:
            viz = MeshcatVisualizer(self.hop_robo.model,
                                    self.hop_robo.visual_model,
                                    self.hop_robo.visual_model)
            viz.viewer = meshcat.Visualizer().open()
            viz.viewer["/Background"].set_property("visible", False)
            viz.viewer["/Grid"].set_property("visible", False)
            viz.viewer["/Axes"].set_property("visible", False)
            viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [0,0,0.5])
            viz.clean()
            viz.loadViewerModel()
            viz.display(q)

        for i in range(simulate_steps):
            current_time = i * self.time_step
            des_pos, des_vel, des_acc = traj_fun(current_time)
            tau_q = self.get_torques(
                des_acc, q, grav_force, total_mass, control_coefficient)
            a = pin.constraintDynamics(self.hop_robo.model, self.hop_robo.data,
                                       q, vq, tau_q,
                                       self.hop_robo.constraint_models,
                                       self.hop_robo.constraint_data,
                                       self.dynamic_settings)

            vq += a * self.time_step
            q = pin.integrate(self.hop_robo.model, q, vq * self.time_step)
            # First coordinate is root_joint
            q_act[i] = q
            vq_act[i] = vq
            acc_act[i] = a
            tau_act[i] = self.generalized_q_to_act_torques(tau_q)

            if is_vis:
                viz.display(q)
        return q_act, vq_act, acc_act, tau_act

    def create_traj_equation(self) -> Callable:
        """Returns function(t) -> (pos, vel, acc)

        Returns:
            Callable: _description_
        """
        final_v = calculate_final_v(self.squat_hop_parameters.hop_flight_hight)
        traj_fun = quartic_func_free_acc(
            self.squat_hop_parameters.squatting_down_hight,
            self.squat_hop_parameters.squatting_up_hight,
            self.squat_hop_parameters.total_time,
            qd0=0,
            qdf=final_v)
        return traj_fun

    def scalar_force_to_wrench(self, force: float):
        """Converts scalar force to 6d vector.

        Args:
            force (float): scalar

        Raises:
            NotImplemented: Only z direction

        Returns:
            _type_: _description_
        """
        if self.squat_hop_parameters.hop_direction == HopDirection.Z:
            ret_val = np.array([0, 0, force, 0, 0, 0])
        else:
            raise NotImplemented
        return ret_val

    def act_torques_to_generalized_q(self, torques: np.ndarray) -> np.ndarray:
        """Converts a actuator size vector 
        to a size vector equal to self.hop_robo.


        Args:
            torques (np.ndarray): _description_

        Returns:
            _type_: q vector self.hop_robo.model.nv
        """
        id_mt1 = self.hop_robo.actuation_model.idqmot[0]
        id_mt2 = self.hop_robo.actuation_model.idqmot[1]
        tau = np.zeros(self.hop_robo.model.nv)
        tau[id_mt1] = torques[0]
        tau[id_mt2] = torques[1]
        return tau

    def generalized_q_to_act_torques(self,
                                     generalized_q: np.ndarray) -> np.ndarray:
        """Converts a size vector equal to self.hop_robo.model.nv 
        to an actuator size vector.

        Args:
            generalized_q (np.ndarray): Vector of generalized coordinates.

        Returns:
            np.ndarray: Vector of torques corresponding to the actuators.
        """
        id_mt1 = self.hop_robo.actuation_model.idqmot[0]
        id_mt2 = self.hop_robo.actuation_model.idqmot[1]
        torques = np.array([generalized_q[id_mt1], generalized_q[id_mt2]])
        return torques

    def get_torques(self,
                    desired_acceleration: float,
                    current_q: np.ndarray,
                    grav_force: float,
                    total_mass: float,
                    control_coefficient: float = 0.8) -> np.ndarray:
        """Calculate actuator torques. With size self.hop_robo.model.nv.

        Args:
            desired_acceleration (float): _description_
            current_q (np.ndarray): _description_
            grav_force (float): _description_
            total_mass (float): Total mass of robot (base + leg)

        Returns:
            np.ndarray: _description_ 
        """
        ground_as_ee_id = self.hop_robo.model.getFrameId(
            self.squat_hop_parameters.ground_link_name)
        ee_id = self.trans_base_robo.model.getFrameId(
            self.squat_hop_parameters.ground_link_name)

        qushka = np.concatenate(
            [np.array([0, 0, current_q[0]]), current_q[1:]])

        # qushka = current_q
        pin.framesForwardKinematics(self.trans_base_robo.model,
                                    self.trans_base_robo.data, qushka)
        pin.forwardKinematics(self.trans_base_robo.model,
                              self.trans_base_robo.data, qushka)
        pin.computeJointJacobians(self.trans_base_robo.model,
                                  self.trans_base_robo.data, qushka)
        J_closed2 = ConstraintFrameJacobian(
            self.trans_base_robo.model,
            self.trans_base_robo.data,
            self.trans_base_robo.constraint_models,
            self.trans_base_robo.constraint_data,
            self.trans_base_robo.actuation_model,
            qushka,
            ee_id,
            self.trans_base_robo.data.oMf[ground_as_ee_id].action
            @ np.zeros(6),
        )

        desired_end_effector_force = control_coefficient * \
            grav_force + total_mass * desired_acceleration
        desired_end_effector_wrench = self.scalar_force_to_wrench(
            desired_end_effector_force)
        desired_q_torques = J_closed2.T @ desired_end_effector_wrench
        tau = self.act_torques_to_generalized_q(desired_q_torques)
        return tau
