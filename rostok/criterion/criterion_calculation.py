from abc import ABC
from typing import Dict, List, Optional, Tuple, Union
from bisect import bisect_left
import numpy as np
import pychrono.core as chrono
from scipy.spatial import distance
from rostok.simulation_chrono.basic_simulation import SimulationResult


#Interface for criterions
class Criterion(ABC):

    def calculate_reward(self, simulation_output:SimulationResult):
        pass


class ForceCriterion(Criterion):

    def calculate_reward(self, simulation_output:SimulationResult) -> float:
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
                if force_module<100:
                    force_modules.append(force_module)

        if len(force_modules) > 0:
            return 1 / (1 + np.mean(np.array(force_modules)))
        else:
            return 0


class TimeCriterion(Criterion):

    def __init__(self, time):
        self.max_simulation_time = time

    def calculate_reward(self, simulation_output:SimulationResult):
        time = simulation_output.time
        if (time) > 0:
            return (time)**2 / (self.max_simulation_time)**2
        else:
            return 0


class ObjectCOGCriterion(Criterion):

    def calculate_reward(self, simulation_output:SimulationResult):
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


class LateForceCriterion(Criterion):

    def __init__(self, cut_off, force_threshold):
        self.cut_off = cut_off
        self.force_threshold = force_threshold

    def calculate_reward(self, simulation_output:SimulationResult):
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
        else: return 0


class LateForceAmountCriterion(Criterion):

    def __init__(self, cut_off):
        self.cut_off = cut_off

    def calculate_reward(self, simulation_output:SimulationResult):
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
        else: return 0

class InstantContactingLinkCriterion(Criterion):
    def __init__(self, time):
        self.reference_time = time

    def calculate_reward(self, simulation_output: SimulationResult):
        time_vector = simulation_output.time_vector
        pos = bisect_left(time_vector, self.reference_time)
        if pos == len(time_vector):
            return 0
        robot_data = simulation_output.robot_final_ds
        body_contacts = robot_data.get_data("n_contacts")
        n_bodies = len(body_contacts.keys())
        contacting_bodies = 0
        for body, contacts in body_contacts.items():
            if contacts[pos] > 0:
                contacting_bodies += 1
        
        return contacting_bodies / n_bodies

class SimulationReward:

    def __init__(self, verbosity = 0) -> None:
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
