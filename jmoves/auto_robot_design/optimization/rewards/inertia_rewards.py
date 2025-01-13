from typing import Tuple

import numpy as np

from auto_robot_design.optimization.rewards.reward_base import Reward
from auto_robot_design.pinokla.calc_criterion import DataDict


class MassReward(Reward):
    """Mass of the robot

    Currently mass reward does not include the base"""

    def __init__(self, mass_key: str) -> None:
        """Set the dictionary keys for the data

        Args:
            mass_key (str): key for the mech mass
        """
        self.mass_key = mass_key
        super().__init__(name='Mass Reward')

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """Just get the total mass from the data dictionaries

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        # get the manipulability for each point at the trajectory
        mass = trajectory_criteria[self.mass_key]
        return -mass, []


class ActuatedMassReward(Reward):
    """Determinant of the mass matrix in coordinates of the actuated joints

    Currently mass reward does not include the base"""

    def __init__(self, mass_key: str, reachability_key: str) -> None:
        """Set the dictionary keys for the data

        Args:
            mass_key (str): key for the mech mass
        """
        self.mass_key = mass_key
        self.reachability_key = reachability_key
        super().__init__(name='Actuated Mass Reward')

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """Calculate mean determinant of the mass matrix in actuated coordinates. 

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
        
        mass = np.linalg.det(point_criteria[self.mass_key])
        return 1/np.mean(mass), mass


class TrajectoryIMFReward(Reward):
    """mean IMF along a trajectory"""

    def __init__(self, imf_key: str, trajectory_key: str, reachability_key: str) -> None:
        """Set the dictionary keys for the data

        Args:
            imf_key (str): key for the value of the IMF
            trajectory_key (str): key for the trajectory points
            error_key (str): key for the pose errors 
        """
        super().__init__(name='Trajectory IMF')
        self.imf_key = imf_key
        self.trajectory_key = trajectory_key
        self.reachability_key = reachability_key

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """Calculate the mean IMF along the trajectory

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

        reward_vector = point_criteria[self.imf_key]
        return np.mean(reward_vector), reward_vector



# This reward is deprecated, in order to use it one have to update both constructor and calculate method
# class EndPointIMFReward(Reward):
#     """IMF in the trajectory edge points"""

#     def __init__(self, imf_key: str, trajectory_key: str, error_key: str) -> None:
#         """Set the dictionary keys for the data

#         Args:
#             imf_key (str): key for the value of the IMF
#             trajectory_key (str): key for the trajectory points
#             error_key (str): key for the pose errors 
#         """
#         self.imf_key = imf_key
#         self.trajectory_key = trajectory_key
#         self.error_key = error_key
#         super().__init__(name='End Point IMF')

#     def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
#         """Calculate the sum of IMF in starting and end points

#         Args:
#             point_criteria (DataDict): all data of the characteristics assigned to each point
#             trajectory_criteria (DataDict): all data of the trajectory characteristics 
#             trajectory_results (DataDict): data of trajectory and trajectory following

#         Returns:
#             float: value of the reward
#         """
#         IMF: list[np.array] = point_criteria[self.imf_key]
#         errors = trajectory_results[self.error_key]
#         if errors[0] > 1e-6:
#             starting_result = 0
#         else:
#             starting_result = IMF[0]

#         if errors[-1] > 1e-6:
#             end_result = 0
#         else:
#             end_result = IMF[-1]

#         return (starting_result + end_result)/2, [starting_result, end_result]
