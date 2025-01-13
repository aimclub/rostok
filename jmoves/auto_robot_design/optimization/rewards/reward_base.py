from operator import itemgetter
from typing import Tuple

import numpy as np

from auto_robot_design.pinokla.calc_criterion import DataDict


class NotReacablePoints(Exception):
    pass

class Reward():
    """Interface for the optimization criteria"""

    def __init__(self, name) -> None:
        self.point_precision = 1e-4
        self.reward_name = name

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """Calculate the value of the criterion from the data"""

        raise NotImplementedError("A reward must implement calculate method!")

    def check_reachability(self, is_reach, checked=True, warning=False):
        """The function that checks the reachability of the mech for all points at trajectory

            The idea is that the trajectory at the moment of the reward calculation is 
            already checked for reachability by the workspace checker.
        """
        if 0 in is_reach and checked:
            if warning:
                print(
                    f'For the reward {self.reward_name} the trajectory has unreachable points with index{np.argmin(is_reach)}')
                return False
            else:
                raise NotReacablePoints(
                    f"All points should be reachable to calculate a reward {self.reward_name}")

        elif 0 in is_reach:
            return False

        return True


class DummyReward(Reward):
    """The reward that can be used for padding."""

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:

        return 0, []


class PositioningReward(Reward):
    """Mean position error for the trajectory"""

    def __init__(self,  pos_error_key: str) -> None:
        """Set the dictionary keys for the data

        Args:
            pos_error_key (str): key for mean position error
        """
        self.pos_error_key = pos_error_key
        super().__init__(name='Trajectory error')

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """Just get the value for the mean positioning error

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        
        mean_error = trajectory_criteria[self.pos_error_key]
        # the empty list is for the consistency with the other rewards
        return -mean_error, []


class PositioningErrorCalculatorOld():
    def __init__(self, error_key):
        self.error_key = error_key
        self.point_threshold = 1e-4

    def calculate(self, trajectory_results: DataDict):
        errors = trajectory_results[self.error_key]
        if np.max(errors) > self.point_threshold:
            # return np.mean(errors)
            return np.max(errors)
        else:
            return 0


class PositioningErrorCalculator():
    """Calculate the special error that that is used as self constrain during optimization
    """

    def __init__(self, jacobian_key, calc_isotropic_thr=True, delta_q_threshold=1):
        self.jacobian_key = jacobian_key
        self.calc_isotropic_thr = calc_isotropic_thr
        self.point_threshold = 1e-4
        self.point_isotropic_threshold = 15
        self.point_isotropic_clip = 3*15
        self.delta_q_threshold = delta_q_threshold

    def calculate(self, trajectory_results_jacob: DataDict, trajectory_results_pos: DataDict):
        """Normalize self.calculate_eig_error and plus self.calculate_pos_error

        Args:
            trajectory_results_jacob (DataDict): _description_
            trajectory_results_pos (DataDict): _description_

        Returns:
            _type_: _description_
        """
        if not np.all(trajectory_results_pos["is_reach"]):
            pos_err = (len(trajectory_results_pos["is_reach"])-np.sum(trajectory_results_pos["is_reach"]))*self.point_threshold
            return pos_err
        else:
            pos_err = 0
        #pos_err = self.calculate_pos_error(trajectory_results_pos)

        #  self.check_continuity(trajectory_results_pos)
        ret = pos_err
        if self.calc_isotropic_thr:
            isotropic_value = self.calculate_eig_error(
                trajectory_results_jacob)
            normalized_isotropic_0_1 = isotropic_value / self.point_isotropic_clip
            isotropic_same_pos_err = (
                normalized_isotropic_0_1*self.point_threshold) / 2
            ret += isotropic_same_pos_err
        return ret

    def calculate_eig_error(self, trajectory_results: DataDict):
        """Return max isotropic clipped by self.point_isotropic_clip

        Args:
            trajectory_results (DataDict): data describing trajectory following

        Returns:
            float: clipped max of the isotropic values
        """
        isotropic_values = self.calculate_isotropic_values(trajectory_results)

        max_isotropic_value = np.max(isotropic_values)
        if max_isotropic_value > self.point_isotropic_threshold:
            clipped_max = np.clip(max_isotropic_value, 0,
                                  self.point_isotropic_clip)
            return clipped_max
        else:
            return 0

    # def check_continuity(self, trajectory_results_pos):
    #     """Check if the difference in angles between two points is less then self.delta_q_threshold radian"""
    #     value = np.max(
    #         np.sum(np.abs(np.diff(trajectory_results_pos['q'], axis=0)), axis=1))
    #     l = len(trajectory_results_pos['q'][0])
    #     if value > self.delta_q_threshold:
    #         #with open('cont_check.txt', 'a') as f:
    #         #f.write(f'Continuity is violated with value: {value}, {l}\n')
    #         pass

    # def calculate_pos_error(self, trajectory_results: DataDict):
    #     """Returns max max value of the errors along trajectory if error at any point exceeds the threshold.

    #     Args:
    #         trajectory_results (DataDict): data describing trajectory following

    #     Returns:
    #         float: max error
    #     """
    #     errors = trajectory_results[self.error_key]
    #     if np.max(errors) > self.point_threshold:
    #         # return np.mean(errors)
    #         return np.max(errors)
    #     else:
    #         return 0

    def calculate_isotropic_values(self, trajectory_results: DataDict) -> np.ndarray:
        """Returns max(eigenvalues) divided by min(eigenvalues) for each jacobian in trajectory_results. 

        Args:
            trajectory_results (DataDict): data describing trajectory following

        Returns:
            np.ndarray: max(eigenvalues)/min(eigenvalues)
        """
        jacobians = trajectory_results[self.jacobian_key]
        isotropic_values = np.zeros(len(jacobians))
        for num, jacob in enumerate(jacobians):
            U, S, Vh = np.linalg.svd(jacob)
            max_eig_val = np.max(S)
            min_eig_val = np.min(S)
            isotropic = max_eig_val / min_eig_val
            isotropic_values[num] = isotropic
        return isotropic_values


class PositioningConstrain():
    """Represents the constrains that are used as a part of the reward function"""
    def __init__(self, error_calculator, points=None) -> None:
        self.points = points
        self.calculator = error_calculator

    def add_points_set(self, points_set):
        """Adds another trajectory for constrain calculation.

        Args:
            points_set (np.array): trajectory description
        """
        if self.points is None:
            self.points = [points_set]
        else:
            self.points.append(points_set)

    def calculate_constrain_error(self, criterion_aggregator, fixed_robot, free_robot):
        """Calculate the constrain error using defined calculator

        Args:
            criterion_aggregator (_type_): _description_
            fixed_robot (_type_): _description_
            free_robot (_type_): _description_

        Returns:
            _type_: _description_
        """
        total_error = 0
        results = []
        for point_set in self.points:
            tmp = criterion_aggregator.get_criteria_data(
                fixed_robot, free_robot, point_set)
            results.append(tmp)
            total_error += self.calculator.calculate(tmp[0], tmp[2])

        return total_error, results


class RewardManager():
    """Manager class to aggregate trajectories and corresponding rewards

        User should add trajectories and then add rewards that are calculated for these trajectories.
    """

    def __init__(self, crag) -> None:
        self.trajectories = {}
        self.rewards = {}
        self.crag = crag
        self.precalculated_trajectories = None
        self.agg_list = []
        self.reward_description = []
        self.trajectory_names = {}

    def add_trajectory(self, trajectory, idx, name="unnamed"):
        if not (idx in self.trajectories):
            self.trajectories[idx] = trajectory
            self.rewards[idx] = []
            self.trajectory_names[idx] = name
        else:
            raise KeyError(
                'Attempt to add trajectory id that already exist in RewardManager')

    def add_reward(self, reward, trajectory_id, weight):
        if trajectory_id in self.trajectories:
            self.rewards[trajectory_id].append((reward, weight))
        else:
            raise KeyError('Trajectory id not in the trajectories dict')

    def add_trajectory_aggregator(self, trajectory_list, agg_type: str):
        if not (agg_type in ['mean', 'median', 'min', 'max']):
            raise ValueError('Wrong aggregation type!')

        if not set(trajectory_list).issubset(set(self.trajectories.keys())):
            raise ValueError('add trajectory before aggregation')

        for lt, _ in self.agg_list:
            if len(set(lt).intersection(set(trajectory_list))) > 0:
                raise ValueError('Each trajectory can be aggregated only once')

        if len(set(map(len, itemgetter(*trajectory_list)(self.rewards)))) > 1:
            raise ValueError(
                'Each trajectory in aggregation must have the same number of rewards')

        self.agg_list.append((trajectory_list, agg_type))

    def close_trajectories(self):
        total_rewards = 0
        exclusion_list = []
        for lst, _ in self.agg_list:
            exclusion_list += lst
            tmp = len(self.rewards[lst[0]])
            self.reward_description.append((lst, tmp))
            total_rewards += tmp

        for idx, rewards in self.rewards.items():
            if idx not in exclusion_list:
                tmp = len(rewards)
                self.reward_description.append((idx, tmp))
                total_rewards += tmp

        return total_rewards

    def calculate_total(self, fixed_robot, free_robot, motor, viz=None):
        # trajectory_rewards = []
        partial_rewards = []
        weighted_partial_rewards = []
        for trajectory_id, trajectory in self.trajectories.items():
            rewards = self.rewards[trajectory_id]
            if self.precalculated_trajectories and (trajectory_id in self.precalculated_trajectories):
                point_criteria_vector, trajectory_criteria, res_dict_fixed = self.precalculated_trajectories[
                    trajectory_id]
            else:
                point_criteria_vector, trajectory_criteria, res_dict_fixed = self.crag.get_criteria_data(
                    fixed_robot, free_robot, trajectory, viz=viz)

            partial_reward = [trajectory_id]
            weighted_partial = [trajectory_id]
            for reward, weight in rewards:
                reward_value = reward.calculate(
                    point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=motor)[0]
                partial_reward.append(reward_value)
                weighted_partial.append(weight * reward_value)
            # update reward lists

            partial_rewards.append(partial_reward)
            weighted_partial_rewards.append(weighted_partial)

        multicriterial_reward = []
        aggregated_partial = []
        exclusion_list = []
        for lst, agg_type in self.agg_list:
            exclusion_list += lst
            local_partial = []
            local_weighted_partial = []
            for v, w in zip(partial_rewards, weighted_partial_rewards):
                if v[0] in lst:
                    local_partial.append(v)
                    local_weighted_partial.append(w)

            tmp_array = np.array(local_partial)
            tmp_array_weighted = np.array(local_weighted_partial)

            if agg_type == 'mean':
                res = np.mean(tmp_array, axis=0)
                res_w = np.mean(local_weighted_partial, axis=0)
            elif agg_type == 'median':
                res = np.median(tmp_array, axis=0)
                res_w = np.median(local_weighted_partial, axis=0)
            elif agg_type == 'min':
                res = np.min(tmp_array, axis=0)
                res_w = np.min(local_weighted_partial, axis=0)
            elif agg_type == 'max':
                res = np.max(tmp_array, axis=0)
                res_w = np.max(local_weighted_partial, axis=0)

            multicriterial_reward += list(res[1::])
            aggregated_partial += list(res_w[1::])

        trajectoryless_partials = []
        for v, w in zip(partial_rewards, weighted_partial_rewards):
            if v[0] not in exclusion_list:
                multicriterial_reward += v[1::]
                aggregated_partial += w[1::]
            trajectoryless_partials += v[1::]

        # calculate the total reward

        total_reward = -np.sum(aggregated_partial)

        return total_reward, trajectoryless_partials, multicriterial_reward

    def dummy_partial(self):
        """Create partial reward with zeros to add for robots that failed constrains"""
        partial_rewards = []
        for trajectory_id, _ in self.trajectories.items():
            rewards = self.rewards[trajectory_id]
            partial_reward = [trajectory_id]
            for _, _ in rewards:
                partial_reward.append(0)
            # Draw trianle if chicken usualy dead
            partial_rewards += partial_reward[1:]
        return partial_rewards

    def check_constrain_trajectory(self, trajectory, results):
        """Checks if a trajectory that was used in constrain calculation is also one of reward trajectories.

            If a trajectory is a reward trajectory save its results and use them to avoid recalculation 
        """
        temp_dict = {}
        for trajectory_id, in_trajectory in self.trajectories.items():
            if np.array_equal(trajectory, in_trajectory):
                temp_dict[trajectory_id] = results

        self.precalculated_trajectories = temp_dict
