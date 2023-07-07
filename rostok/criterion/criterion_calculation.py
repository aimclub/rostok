from abc import ABC
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pychrono.core as chrono
from scipy.spatial import distance

from rostok.simulation_chrono.basic_simulation import SimulationResult
from rostok.criterion.simulation_flags import SimulationSingleEvent, EventContactTimeOut,EventGrasp, EventSlipOut


#Interface for criterions
class Criterion(ABC):

    def calculate_reward(self, simulation_output: SimulationResult):
        pass

class TimeCriterion(Criterion):

    def __init__(self, time:float, event_timeout:EventContactTimeOut):
        self.max_simulation_time = time
        self.event_timeout = event_timeout

    def calculate_reward(self, simulation_output: SimulationResult):
        if self.event_timeout.state:
            return 0
        else:
            time = simulation_output.time
            if (time) > 0:
                return (time)**2 / (self.max_simulation_time)**2
            else:
                return 0

class ForceCriterion(Criterion):

    def __init__(self, time:float, event_timeout:EventContactTimeOut):
        self.max_simulation_time = time
        self.event_timeout = event_timeout

    def calculate_reward(self, simulation_output: SimulationResult) -> float:
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
    def __init__(self, grasp_event:EventGrasp):
        self.grasp_event = grasp_event

    def calculate_reward(self, simulation_output: SimulationResult):
        if self.grasp_event.state:
            env_data = simulation_output.environment_final_ds
            body_COG = env_data.get_data("COG")[0][self.grasp_event.step_n]
            body_outer_force_center = env_data.get_data("force_center")[0][self.grasp_event.step_n]
            dist = distance.euclidean(body_COG, body_outer_force_center)
            return 1/(1+dist)
        else:
            return 0

class InstantForceCriterion(Criterion):

    def __init__(self, grasp_event:EventGrasp):
        self.grasp_event = grasp_event

    def calculate_reward(self, simulation_output: SimulationResult):
        if self.grasp_event.state:
            env_data = simulation_output.environment_final_ds
            body_contacts: np.ndarray = env_data.get_data("forces")[0][self.grasp_event.step_n]
            if body_contacts > 0:
                forces = []
                for force in body_contacts:
                    forces.append(np.linalg.norm(force))

                return 1 / (1 + np.std(forces))
            else:
                return 0
        else:
            return 0

class InstantContactingLinkCriterion(Criterion):

    def __init__(self, grasp_event:EventGrasp):
        self.grasp_event = grasp_event

    def calculate_reward(self, simulation_output: SimulationResult):
        if self.grasp_event.state:
            robot_data = simulation_output.robot_final_ds
            robot_contacts = robot_data.get_data("n_contacts")
            n_bodies = len(robot_contacts.keys())
            contacting_bodies = 0
            for body, contacts in robot_contacts.items():
                if contacts[self.grasp_event.step_n] > 0:
                    contacting_bodies += 1

            return contacting_bodies / n_bodies
        else:
            return 0

class GraspTimeCriterion(Criterion):
    def __init__(self, grasp_event:EventGrasp, total_steps:int):
        self.grasp_event = grasp_event
        self.total_steps = total_steps

    def calculate_reward(self, simulation_output: SimulationResult):
        if self.grasp_event.state:
            return (self.total_steps -  self.grasp_event.step_n)/self.total_steps
        else:
            return 0

class FinalPositionCriterion(Criterion):
    def __init__(self, reference_distance:float, grasp_event:EventGrasp, slipout_event:EventSlipOut):
        self.reference_distance = reference_distance
        self.grasp_event = grasp_event
        self.slipout_event = slipout_event

    def calculate_reward(self, simulation_output: SimulationResult):
        if self.grasp_event.state and not self.slipout_event:
            env_data = simulation_output.environment_final_ds
            grasp_pos = env_data.get_data("COG")[0][self.grasp_event.step_n]
            final_pos = env_data.get_data("COG")[0][-1]
            dist = distance.euclidean(grasp_pos, final_pos)
            if dist < self.reference_distance:
                

        else: 
            return 0






class ObjectCOGCriterion(Criterion):

    def calculate_reward(self, simulation_output: SimulationResult):
        dist_list = []
        env_data = simulation_output.environment_final_ds
        body_COG = env_data.get_data("COG")[0]  # List[Tuple[step_n, List[x,y,z]]]
        body_outer_force_center = env_data.get_data("force_center")[0]
        dist_list = []
        for cog, force in zip(body_COG, body_outer_force_center):
            if not force is np.nan:
                dist_list.append(distance.euclidean(cog, force))

        if np.size(dist_list) > 0:
            cog_crit = 1 / (1 + np.mean(dist_list))
        else:
            cog_crit = 0

        return cog_crit


class SimulationReward:

    def __init__(self, verbosity=0) -> None:
        self.criteria: List[Criterion] = []
        self.weights: List[float] = []
        self.verbosity = verbosity

    def add_criterion(self, citerion: Criterion, weight: float):
        self.criteria.append(citerion)
        self.weights.append(weight)

    def calculate_reward(self, simulation_output):
        partial_rewards = []
        for criterion in self.criteria:
            partial_rewards.append(criterion.calculate_reward(simulation_output))

        if self.verbosity > 0:
            print([round(x, 3) for x in partial_rewards])

        total_reward = sum([a * b for a, b in zip(partial_rewards, self.weights)])
        return round(total_reward, 3)


"""class LateForceCriterion(Criterion):

    def __init__(self, cut_off, force_threshold):
        self.cut_off = cut_off
        self.force_threshold = force_threshold

    def calculate_reward(self, simulation_output: SimulationResult):
        env_data = simulation_output.environment_final_ds
        body_contacts = env_data.get_data("forces")[0]
        step_cutoff = int(simulation_output.n_steps * self.cut_off)
        body_contacts_cut = body_contacts[step_cutoff::]
        counter = 0
        for data in body_contacts_cut:
            total_force_module = 0
            for contact in data:
                total_force_module += np.linalg.norm(contact[1])

            if total_force_module > self.force_threshold:
                counter += 1

        if len(body_contacts_cut) > 0:
            return counter / (len(body_contacts_cut))
        else:
            return 0


class LateForceAmountCriterion(Criterion):

    def __init__(self, cut_off):
        self.cut_off = cut_off

    def calculate_reward(self, simulation_output: SimulationResult):
        env_data = simulation_output.environment_final_ds
        body_contacts = env_data.get_data("n_contacts")[0]
        step_cutoff = int(simulation_output.n_steps * self.cut_off)
        counter = 0
        body_contacts_cut = body_contacts[step_cutoff::]
        for data in body_contacts_cut:
            if not data is np.nan:
                counter += data

        if len(body_contacts_cut) > 0:
            return counter / (len(body_contacts_cut))
        else:
            return 0








class SimulationFlagBasedReward(SimulationReward):

    def __init__(self, verbosity=0):
        super().__init__(verbosity)
        self.flag_lists = []

    def add_criterion(self, citerion: Criterion, weight: float, flags: List[FlagSimulation] = []):
        self.criteria.append(citerion)
        self.weights.append(weight)
        self.flag_lists.append(flags)

    def calculate_reward(self, simulation_output):
        partial_rewards = []
        for criterion, flags in zip(self.criteria, self.flag_lists):
            if any(flags):
                partial_rewards.append(0)
            else:
                partial_rewards.append(criterion.calculate_reward(simulation_output))

        if self.verbosity > 0:
            print([round(x, 3) for x in partial_rewards])

        total_reward = sum([a * b for a, b in zip(partial_rewards, self.weights)])
        return round(total_reward, 3)
"""