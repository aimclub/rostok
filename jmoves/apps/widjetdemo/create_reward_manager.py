from networkx import Graph
import numpy as np


from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.optimization.rewards.reward_base import PositioningConstrain, PositioningErrorCalculator, RewardManager


def get_manager_mock(workspace_traj: np.ndarray):
    """Returns fake args for CalculateMultiCriteriaProblem.


    Args:
        workspace_traj (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    dict_trajectory_criteria = {

    }
    # criteria calculated for each point on the trajectory
    dict_point_criteria = {

    }

    crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
    error_calculator = PositioningErrorCalculator(
        error_key='error', jacobian_key="Manip_Jacobian")
    soft_constrain = PositioningConstrain(
        error_calculator=error_calculator, points=[workspace_traj])
    reward_manager = RewardManager(crag=crag)

    return reward_manager, crag, soft_constrain
