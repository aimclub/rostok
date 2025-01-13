import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from auto_robot_design.pinokla.closed_loop_jacobian import (
    constraint_jacobian_active_to_passive,
    ConstraintFrameJacobian,
    jacobian_constraint,
)


class TorqueComputedControl:
    def __init__(
            self, robot, Kp: np.ndarray, Kd: np.ndarray, use_dJdt_term: bool = False
        ):
            """
            Initialize class for computed torque control for a robot.

            Args:
                robot: The robot model.
                Kp: The proportional gain matrix of shape (nmot, nmot).
                Kd: The derivative gain matrix of shape (nmot, nmot).
                use_dJdt_term: A boolean indicating whether to use the dJ/dt term in the control law.
            """

            self.robot = robot
            self.use_dJdt = use_dJdt_term

            nmot = len(robot.actuation_model.idqmot)
            assert Kp.shape == (nmot, nmot)
            assert Kd.shape == (nmot, nmot)

            self.Kp = Kp
            self.Kd = Kd

            self.ids_mot = robot.actuation_model.idqmot
            self.ids_vmot = robot.actuation_model.idvmot

            self.ids_free = self.robot.actuation_model.idqfree
            self.ids_vfree = self.robot.actuation_model.idvfree

            self.tauq = np.zeros(robot.model.nv)

            self.prev_Jmot = np.zeros((len(self.ids_vfree), len(self.ids_vmot)))
            self.prev_Jfree = np.zeros((len(self.ids_vfree), len(self.ids_vfree)))

    def compute(
        self,
        q: np.ndarray,
        vq: np.ndarray,
        q_a_ref: np.ndarray,
        vq_a_ref: np.ndarray,
        ddq_a_ref: np.ndarray,
    ):
        """
        Compute the control torque for the robot based on the given inputs.

        Args:
            q (np.ndarray): The joint positions of the robot.
            vq (np.ndarray): The joint velocities of the robot.
            q_a_ref (np.ndarray): The desired joint positions for the active joints.
            vq_a_ref (np.ndarray): The desired joint velocities for the active joints.
            ddq_a_ref (np.ndarray): The desired joint accelerations for the active joints.

        Returns:
            np.ndarray: The control torque for the robot.

        """
        if self.use_dJdt:
            Jmot, Jfree = jacobian_constraint(
                self.robot.model,
                self.robot.data,
                self.robot.constraint_models,
                self.robot.constraint_data,
                self.robot.actuation_model,
                q,
            )
            epsilon = 1e-6
            Jmot_ = np.zeros((self.robot.nq, Jmot.shape[0], Jmot.shape[1]))
            Jfree_ = np.zeros((self.robot.nq, Jfree.shape[0], Jfree.shape[1]))
            for i in range(len(self.robot.nq)):
                q_ = q.copy()
                q_[i] += epsilon
                Jmot_i, Jfree_i = jacobian_constraint(
                    self.robot.model,
                    self.robot.data,
                    self.robot.constraint_models,
                    self.robot.constraint_data,
                    self.robot.actuation_model,
                    q_,
                )
                Jmot_[i, :, :] = Jmot_i
                Jfree_[i, :, :] = Jfree_i
            d_Jmot = np.dot(Jmot_, vq)
            d_Jfree = np.dot(Jfree_, vq)
            # d_Jmot = ((Jmot - self.prev_Jmot) / DT).round(6)
            # d_Jfree = ((Jfree - self.prev_Jfree) / DT).round(6)
            a_d = -np.linalg.pinv(Jfree) @ (
                d_Jmot @ vq[self.ids_vmot] + d_Jfree @ vq[self.ids_vfree]
            )
        else:
            a_d = np.zeros(len(self.ids_vfree))

        q_a = q[self.ids_mot]
        vq_a = vq[self.ids_vmot]

        M = pin.crba(self.robot.model, self.robot.data, q)
        g = pin.computeGeneralizedGravity(self.robot.model, self.robot.data, q)
        C = pin.computeCoriolisMatrix(self.robot.model, self.robot.data, q, vq)

        Jda, E_tau = constraint_jacobian_active_to_passive(
            self.robot.model,
            self.robot.data,
            self.robot.constraint_models,
            self.robot.constraint_data,
            self.robot.actuation_model,
            q,
        )

        Ma = Jda.T @ E_tau.T @ M @ E_tau @ Jda
        ga = Jda.T @ E_tau.T @ g
        Ca = Jda.T @ E_tau.T @ C @ E_tau @ Jda

        tau_a = (
            Ma @ ddq_a_ref
            + Ca @ vq_a_ref
            + ga
            - Ma @ self.Kp @ (q_a - q_a_ref)
            - Ma @ self.Kd @ (vq_a - vq_a_ref)
        )
        self.tauq[self.ids_mot] = tau_a

        return self.tauq


class OperationSpacePDControl:
    def __init__(
        self, robot, Kp: np.ndarray, Kd: np.ndarray, id_frame_end_effector: int
    ):
        """
        Initialize class for operation space PD control for a robot.

        Args:
            robot: The robot model.
            Kp: The proportional gain matrix of shape (6, 6).
            Kd: The derivative gain matrix of shape (6, 6).
            id_frame_end_effector: The ID of the end effector frame.

        Returns:
            None
        """

        self.robot = robot
        self.id_frame = id_frame_end_effector

        assert Kp.shape == (6, 6)
        assert Kd.shape == (6, 6)

        self.Kp = Kp
        self.Kd = Kd

        self.ids_mot = robot.actuation_model.idqmot
        self.ids_vmot = robot.actuation_model.idvmot

        self.ids_free = self.robot.actuation_model.idqfree
        self.ids_vfree = self.robot.actuation_model.idvfree

        self.tauq = np.zeros(robot.model.nv)

    def compute(self, q: np.ndarray, vq, x_ref: np.ndarray, dx_ref: np.ndarray):
        """
        Compute the control input for the robot based on the given state and reference values.

        Args:
            q (np.ndarray): The joint positions of the robot.
            vq: The joint velocities of the robot.
            x_ref (np.ndarray): The desired reference state.
            dx_ref (np.ndarray): The desired reference state velocity.

        Returns:
            np.ndarray: The computed control input for the robot.

        """
        J_closed = ConstraintFrameJacobian(
            self.robot.model,
            self.robot.data,
            self.robot.constraint_models,
            self.robot.constraint_data,
            self.robot.actuation_model,
            q,
            self.id_frame,
            self.robot.data.oMf[self.id_frame].action @ dx_ref,
        )
        x_body_curr = np.concatenate(
            (
                self.robot.data.oMf[self.id_frame].translation,
                R.from_matrix(self.robot.data.oMf[self.id_frame].rotation).as_rotvec(),
            )
        )
        # vq_a_ref = vq_cstr[self.ids_vmot]
        Jda, __ = constraint_jacobian_active_to_passive(
            self.robot.model,
            self.robot.data,
            self.robot.constraint_models,
            self.robot.constraint_data,
            self.robot.actuation_model,
            q,
            
        )

        g = pin.computeGeneralizedGravity(self.robot.model, self.robot.data, q)

        vq_a = vq[self.ids_vmot]

        # first = self.Kp @ (x_ref - x_body_curr)
        # second = self.Kd @ (dx_ref - J_closed @ vq_a)
        # third = J_closed.T@(first + second)
        # tau_a = third + Jda.T @ E_tau.T @ g
        tau_a = (
            J_closed.T
            @ (self.Kp @ (x_ref - x_body_curr) + self.Kd @ (dx_ref - J_closed @ vq_a))
            + Jda.T @ g
        )

        self.tauq[self.ids_mot] = tau_a

        return self.tauq
