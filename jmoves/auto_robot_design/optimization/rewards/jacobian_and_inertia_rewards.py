from typing import Tuple

import numpy as np

from auto_robot_design.optimization.rewards.reward_base import Reward
from auto_robot_design.pinokla.calc_criterion import DataDict

GRAVITY = 9.81


def calculate_achievable_forces_z(manipulability_matrices: list[np.array], pick_effort: float,
                                  max_effort_coefficient: float) -> np.ndarray:

    n_steps = len(manipulability_matrices)

    achievable_forces_z = np.zeros(n_steps)
    for i in range(n_steps):
        # the force matrix is the transpose of Jacobian, it transforms forces into torques
        force_matrix = np.transpose(manipulability_matrices[i])
        # calculate torque vector that is required to get unit force in the z direction.
        # it also declares the ratio of torques that provides z-directed force
        z_unit_force_torques = np.abs(force_matrix @ np.array([0, 1]))
        # calculate the factor that max out the higher torque
        achievable_force_z = pick_effort * \
            max_effort_coefficient/max(z_unit_force_torques)
        # calculate extra force that can be applied to the payload
        achievable_forces_z[i] = abs(achievable_force_z)

    return achievable_forces_z


class HeavyLiftingReward(Reward):
    """Calculate the mass that can be held still using up to max_effort_coef of the motor capacity.

    Final reward is the minimum mass that can be held still at each point at the trajectory.

    Args:
        Reward (float): mass capacity
    """

    def __init__(self,
                 manipulability_key,
                 mass_key: str,
                 reachability_key: str,
                 max_effort_coef=0.7) -> None:
        super().__init__(name="Heavy Lifting Reward")
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.reachability_key = reachability_key
        self.mass_key = mass_key

    def calculate(self, point_criteria: DataDict,
                  trajectory_criteria: DataDict, trajectory_results: DataDict,
                  **kwargs) -> Tuple[float, list[float]]:
        """The reward is the minimum mass that can be held still using up to max_effort_coef of the motor capacity at the trajectory.

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Raises:
            KeyError: this function requires motor description

        Returns:
            float: value of the reward
            list[float]: value of the mass that can be held still at each point 
        """
        if "Actuator" in kwargs:
            pick_effort = kwargs["Actuator"].peak_effort
        else:
            raise KeyError("Lifting criterion requires the Actuator")

        is_reached = trajectory_results[self.reachability_key]
        is_trajectory_reachable = self.check_reachability(is_reached)
        # the reward is none zero only if all points are reached
        if not is_trajectory_reachable:
            return 0, []
        # manipulability is the jacobian between the actuators and endeffector
        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        mass = trajectory_criteria[self.mass_key]
        achievable_forces_z_vec = calculate_achievable_forces_z(manipulability_matrices, pick_effort, self.max_effort_coefficient)
        reward_vector = achievable_forces_z_vec / (GRAVITY * mass)
        reward = np.min(reward_vector)

        return reward, reward_vector


class MeanHeavyLiftingReward(Reward):
    """Calculate the mass that can be held still using up to 70% of the motor capacity.

        Final reward is the mean of the mass that can be held still at each point.

    Args:
        Reward (float): mass capacity
    """

    def __init__(self, manipulability_key, mass_key: str, reachability_key: str, max_effort_coef=0.7) -> None:
        super().__init__('Mean Heavy Lifting Reward')
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.reachability_key = reachability_key
        self.mass_key = mass_key

    def calculate(self, point_criteria: DataDict,
                  trajectory_criteria: DataDict, trajectory_results: DataDict,
                  **kwargs) -> Tuple[float, list[float]]:
        """_summary_

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Raises:
            KeyError: this function requires motor description

        Returns:
            float: value of the reward
        """
        if "Actuator" in kwargs:
            pick_effort = kwargs["Actuator"].peak_effort
        else:
            raise KeyError("Lifting criterion requires the Actuator")

        is_reached = trajectory_results[self.reachability_key]
        is_trajectory_reachable = self.check_reachability(is_reached)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []

        manipulability_matrices: list[np.array] = point_criteria[
            self.manip_key]
        mass = trajectory_criteria[self.mass_key]
        achievable_forces_z_vec = calculate_achievable_forces_z(
            manipulability_matrices, pick_effort, self.max_effort_coefficient)
        reward_vector = achievable_forces_z_vec / (GRAVITY * mass)
        reward = np.mean(reward_vector)

        return reward, reward_vector


class AccelerationCapability(Reward):
    """Calculate the reward that combine effective inertia and force capability. 

        At a point it is an acceleration along the trajectory the EE would have in zero gravity if it has zero speed.
        The final reward is the mean of the acceleration at each point.
    """

    def __init__(self, manipulability_key: str, trajectory_key: str, reachability_key: str, actuated_mass_key: str, max_effort_coef=0.7) -> None:
        super().__init__('Acceleration Capability')
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.reachability_key = reachability_key
        self.actuated_mass_key = actuated_mass_key

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """_summary_

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        if "Actuator" in kwargs:
            pick_effort = kwargs["Actuator"].peak_effort
        else:
            raise KeyError("Lifting criterion requires the Actuator")

        is_reached = trajectory_results[self.reachability_key]
        is_trajectory_reachable = self.check_reachability(is_reached)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []
        # manipulability is the jacobian between the actuators and endeffector
        # get the manipulability for each point at the trajectory
        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        effective_mass_matrices: list[np.array] = point_criteria[self.actuated_mass_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        # we just get the vector from current point to the next at each point
        diff_vector = np.diff(trajectory_points, axis=0)[:, [0, 2]]
        n_steps = len(trajectory_points)
        reward_vector =np.zeros(n_steps-1)# reward does not exist for the last point
        for i in range(n_steps-1):
            # get the direction of the trajectory
            trajectory_shift = diff_vector[i]
            trajectory_direction = trajectory_shift / \
                np.linalg.norm(trajectory_shift)

            # get the manipulability matrix and mass matrix for the current point
            manipulability_matrix: np.array = manipulability_matrices[i]
            effective_mass_matrix: np.array = effective_mass_matrices[i]
            # calculate the matrix that transforms quasi-static acceleration to required torque
            acc_2_torque = effective_mass_matrix@np.linalg.inv(
                manipulability_matrix)
            # calculate the torque vector that provides the unit acceleration in the direction of the trajectory
            unit_acc_torque = np.abs(acc_2_torque@trajectory_direction)
            # calculate the factor that max out the higher torque
            acc = pick_effort*self.max_effort_coefficient/max(unit_acc_torque)
            reward_vector[i] = acc

        return np.mean(reward_vector), reward_vector


class MinAccelerationCapability(Reward):
    """Calculate the reward that combine effective inertia and force capability.

        Final reward is the mean value of the minimum singular value of the manipulability matrix at each point.
    """

    def __init__(self,
                 manipulability_key: str,
                 trajectory_key: str,
                 reachability_key: str,
                 actuated_mass_key: str,
                 max_effort_coef=0.7) -> None:
        super().__init__(name='Min Acceleration Capability')
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.reachability_key = reachability_key
        self.actuated_mass_key = actuated_mass_key

    def calculate(self, point_criteria: DataDict,
                  trajectory_criteria: DataDict, trajectory_results: DataDict,
                  **kwargs) -> Tuple[float, list[float]]:
        """_summary_

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """

        is_reached = trajectory_results[self.reachability_key]
        is_trajectory_reachable = self.check_reachability(is_reached)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []

        # get the manipulability for each point at the trajectory
        manipulability_matrices: list[np.array] = point_criteria[
            self.manip_key]
        effective_mass_matrices: list[np.array] = point_criteria[
            self.actuated_mass_key]

        n_steps = len(is_reached)
        reward_vector = np.zeros(n_steps)
        for i in range(n_steps):
            # get the manipulability matrix and mass matrix for the current point
            manipulability_matrix: np.array = manipulability_matrices[i]
            effective_mass_matrix: np.array = effective_mass_matrices[i]
            # calculate the matrix that transforms quasi-static acceleration to required torque

            torque_2_acc = manipulability_matrix @ np.linalg.inv(
                effective_mass_matrix)
            step_result = np.min(abs(np.linalg.svd(torque_2_acc, compute_uv=False)))
            reward_vector[i] = step_result

        return np.mean(reward_vector), reward_vector
