from abc import ABC
import json
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pychrono.core as chrono
from scipy.spatial import distance

from rostok.simulation_chrono.simulation_utils import SimulationResult
from rostok.criterion.simulation_flags import SimulationSingleEvent, EventContactTimeOut, EventGrasp, EventSlipOut
from rostok.utils.json_encoder import RostokJSONEncoder


#Interface for criterions
class Criterion(ABC):

    def calculate_reward(self, simulation_output: SimulationResult):
        """The function that returns the reward as a result of the simulation.

        Args:
            simulation_output (SimulationResult): the result of the simulation  
        """
        pass

    def __repr__(self) -> str:
        json_data = json.dumps(self, cls=RostokJSONEncoder)
        return json_data

    def __str__(self) -> str:
        json_data = json.dumps(self, indent=4, cls=RostokJSONEncoder)
        return json_data


class TimeCriterion(Criterion):
    """Reward based on the time simulation if the grasp doesn't happen.

        Attributes:
            max_simulation_time (float): maximum simulation time 
            event_timeout (EventContactTimeOut): event of contact time out
            event_grasp (EventGrasp): event of object grasping 
    """

    def __init__(self, time: float, event_timeout: EventContactTimeOut, event_grasp: EventGrasp):
        self.max_simulation_time = time
        self.event_timeout = event_timeout
        self.event_grasp = event_grasp

    def calculate_reward(self, simulation_output: SimulationResult):
        """Return 1 for all robots that can grasp the object. 
            Return 0 if there is no contact. If the robot only touches the object and 
            than lose contact, return reward that increases with time of the contact 

        Returns:
            float: reward of the criterion
        """
        if self.event_grasp.state:
            return 1

        if self.event_timeout.state:
            return 0
        else:
            time = simulation_output.time
            if (time) > 0:
                return (time)**2 / (self.max_simulation_time)**2
            else:
                return 0


class ForceCriterion(Criterion):
    """Reward based on the mean force of the robot and object contact.

        Attributes:
            event_timeout (EventContactTimeOut): event of contact time out
    """

    def __init__(self, event_timeout: EventContactTimeOut):
        self.event_timeout = event_timeout

    def calculate_reward(self, simulation_output: SimulationResult) -> float:
        """Return 0 if there is no contact. For every step where object is in contact with robot
            the total force is added to a list and the final reward is calculated using the mean 
            value of that list

        Returns:
            float: reward of the criterion
        """

        if self.event_timeout.state:
            return 0
        else:
            env_data = simulation_output.environment_final_ds
            body_contacts: List[np.ndarray] = env_data.get_data("forces")[0]
            force_modules = []
            for data in body_contacts:
                if data.size > 0:
                    total_force = np.zeros(3)
                    for force in data:
                        total_force += force[1]

                    force_module = np.linalg.norm(total_force)
                    # Cut the steps with huge forces
                    if force_module < 100:
                        force_modules.append(force_module)

            if len(force_modules) > 0:
                return 1 / (1 + np.mean(np.array(force_modules)))
            else:
                return 0


class InstantObjectCOGCriterion(Criterion):
    """Reward based on the distance between object COG and force centroid.

        Attributes:
            event_grasp (EventGrasp): event of object grasping 
    """

    def __init__(self, grasp_event: EventGrasp):
        self.grasp_event = grasp_event

    def calculate_reward(self, simulation_output: SimulationResult):
        """Calculate the reward based on distance between object COG and force centroid in the moment of grasp.

        Returns:
            float: reward of the criterion
        """
        if self.grasp_event.state:
            env_data = simulation_output.environment_final_ds
            body_COG = env_data.get_data("COG")[0][self.grasp_event.step_n + 1]
            body_outer_force_center = env_data.get_data("force_center")[0][self.grasp_event.step_n +
                                                                           1]
            if body_outer_force_center is np.nan:
                print(body_COG, body_outer_force_center)
            dist = distance.euclidean(body_COG, body_outer_force_center)
            return 1 / (1 + dist)
        else:
            return 0


class InstantForceCriterion(Criterion):
    """Criterion based on the std of force modules.

        Attributes:
            event_grasp (EventGrasp): event of object grasping 
    """

    def __init__(self, grasp_event: EventGrasp):
        self.grasp_event = grasp_event

    def calculate_reward(self, simulation_output: SimulationResult):
        """Calculate std of force modules in the grasp moment and calculate reward using it. 

        Returns:
            float: reward of the criterion
        """
        if self.grasp_event.state:
            env_data = simulation_output.environment_final_ds
            body_contacts: np.ndarray = env_data.get_data("forces")[0][self.grasp_event.step_n + 1]
            if len(body_contacts) > 0:
                forces = []
                for force in body_contacts:
                    forces.append(np.linalg.norm(force))

                return 1 / (1 + np.std(forces))
            else:
                return 0
        else:
            return 0


class InstantContactingLinkCriterion(Criterion):
    """Criterion based on the percentage of contacting links.

        Attributes:
            event_grasp (EventGrasp): event of object grasping 
    """

    def __init__(self, grasp_event: EventGrasp):
        self.grasp_event = grasp_event

    def calculate_reward(self, simulation_output: SimulationResult):
        """The reward is the fraction of the links that contacts with object in the grasp moment. 

        Returns:
            float: reward of the criterion
        """
        if self.grasp_event.state:
            robot_data = simulation_output.robot_final_ds
            robot_contacts = robot_data.get_data("n_contacts")
            n_bodies = len(robot_contacts.keys())
            contacting_bodies = 0
            for _, contacts in robot_contacts.items():
                if contacts[self.grasp_event.step_n + 1] > 0:
                    contacting_bodies += 1

            return contacting_bodies / n_bodies
        else:
            return 0


class GraspTimeCriterion(Criterion):
    """Criterion based on the time before grasp.

        Attributes:
            event_grasp (EventGrasp): event of object grasping
            total_steps (total_steps): the amount of the possible steps
    """

    def __init__(self, grasp_event: EventGrasp, total_steps: int):
        self.grasp_event = grasp_event
        self.total_steps = total_steps

    def calculate_reward(self, simulation_output: SimulationResult):
        """Reward depends on the speed of the grasp. 

        Returns:
            float: reward of the criterion
        """
        if self.grasp_event.state:
            return (self.total_steps - self.grasp_event.step_n) / self.total_steps
        else:
            return 0


class FinalPositionCriterion(Criterion):
    """Criterion based on the position change after applying testing force. 

        Attributes:
            reference_distance (float): reference distance for the criterion
            grasp_event (EventGrasp): event of object grasping
            slipout_event (EventSlipOut): event of object slip out
    """

    def __init__(self, reference_distance: float, grasp_event: EventGrasp,
                 slipout_event: EventSlipOut):
        self.reference_distance = reference_distance
        self.grasp_event = grasp_event
        self.slipout_event = slipout_event

    def calculate_reward(self, simulation_output: SimulationResult):
        """The reward is 1 - dist(position in the grasp moment, position in the final moment)/(reference distance)

        Returns:
            float: reward of the criterion
        """
        if self.grasp_event.state and not self.slipout_event.state:
            env_data = simulation_output.environment_final_ds
            grasp_pos = env_data.get_data("COG")[0][self.grasp_event.step_n + 1]
            final_pos = env_data.get_data("COG")[0][-1]
            dist = distance.euclidean(grasp_pos, final_pos)
            if dist <= self.reference_distance:
                return 1 - dist / self.reference_distance
            else:
                return 0

        else:
            return 0


class SimulationReward:
    """Aggregate criterions and weights to calculate reward.

        Attributes:
            criteria (List[Criterion]): list of criterions
            weights (List[float]): criterion weights
            verbosity (int): parameter to control console output
    """

    def __init__(self, verbosity=0) -> None:
        self.criteria: List[Criterion] = []
        self.weights: List[float] = []
        self.verbosity = verbosity

    def add_criterion(self, citerion: Criterion, weight: float):
        """Add criterion and weight to the lists.

        Args:
            citerion (Criterion): new criterion
            weight (float): weight of the new criterion
        """
        self.criteria.append(citerion)
        self.weights.append(weight)

    def calculate_reward(self, simulation_output, partial=False):
        """Calculate all rewards and return weighted sum of them.

        Args:
            simulation_output (_type_): the results of the simulation

        Returns:
            float: total reward
        """
        partial_rewards = []
        for criterion in self.criteria:
            reward = criterion.calculate_reward(simulation_output)
            partial_rewards.append(round(reward,3))

        if partial:
            return partial_rewards
        if self.verbosity > 0:
            print([round(x, 3) for x in partial_rewards])

        total_reward = sum([a * b for a, b in zip(partial_rewards, self.weights)])
        
        if np.isclose(total_reward, 0, atol=1e-3):
            total_reward = 0.02
        return round(total_reward, 3)

    def __repr__(self) -> str:
        json_data = json.dumps(self, cls=RostokJSONEncoder)
        return json_data

    def __str__(self) -> str:
        json_data = json.dumps(self, indent=4, cls=RostokJSONEncoder)
        return json_data
