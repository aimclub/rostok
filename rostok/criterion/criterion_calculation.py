from abc import ABC
from typing import Dict, List, Optional, Tuple, Union
import pychrono.core as chrono
import numpy as np
from scipy.spatial import distance


#Interface for criterions
class Criterion(ABC):

    def calculate_reward(self, simulation_output):
        pass


class ForceCriterion(Criterion):

    def calculate_reward(self, simulation_output):
        env_data = simulation_output[1]
        # List[Tuple[step_n,List[Tuple[Coordinates, Force]]]]
        body_contacts = env_data.get_data("forces")[0]  
        force_modules = []
        for data in body_contacts:
            if len(data[1]) > 0:
                total_force = chrono.ChVectorD()
                for force in data[1]:
                    total_force+=force[1]

                force_modules.append(total_force.Length())

        if len(force_modules) > 0:
            return 1 / (1 + np.mean(np.array(force_modules)))
        else:
            return 0


class TimeCriterion(Criterion):
    def __init__(self, time):
        self.max_simulation_time = time

    def calculate_reward(self, simulation_output):
        if (simulation_output[0]) > 0:
            return (simulation_output[0])**2 / (self.max_simulation_time)**2
        else:
            return 0


class ObjectCOGCriterion(Criterion):

    def calculate_reward(self, simulation_output):
        dist_list = []
        env_data = simulation_output[1]
        body_COG = env_data.get_data("COG")[0]  # List[Tuple[step_n, List[x,y,z]]]
        body_outer_force_center = env_data.get_data("force_center")[
            0]  # List[Tuple[step_n,List[x,y,z]]]
        dist_list = []
        for cog, force in zip(body_COG, body_outer_force_center):
            if force[1]:
                dist_list.append(distance.euclidean(np.array(cog[1]), np.array(force[1])))

        if np.size(dist_list) > 0:
            cog_crit = 1 / (1 + np.mean(dist_list))
        else:
            cog_crit = 0

        return cog_crit

class LateForceCriterion(Criterion):
    def __init__(self, total_steps, cut_off, force_threshold):
        self.total_steps = total_steps
        self.cut_off = cut_off
        self.force_threshold = force_threshold

    def calculate_reward(self, simulation_output):
        env_data = simulation_output[1]
        body_contacts = env_data.get_data("forces")[0]
        step_cutoff = int(self.total_steps * self.cut_off) 
        counter = 0
        for data in body_contacts:
            if data[0] < step_cutoff:
                continue
            else:
                total_force_module = 0
                for contact in data[1]:
                    total_force_module+=contact[1].Length()
                if total_force_module > self.force_threshold:
                    counter+=1

        return counter/ (self.total_steps-step_cutoff)
    
class LateForceAmountCriterion(Criterion):
    def __init__(self, total_steps, cut_off):
        self.total_steps = total_steps
        self.cut_off = cut_off

    def calculate_reward(self, simulation_output):
        env_data = simulation_output[1]
        body_contacts = env_data.get_data("n_contacts")[0]
        step_cutoff = int(self.total_steps * self.cut_off) 
        counter = 0
        for data in body_contacts:
            if data[0] < step_cutoff:
                continue
            else:
                counter += data[1]

        return counter/ (self.total_steps-step_cutoff)


class SimulationReward:

    def __init__(self) -> None:
        self.criteria: List[Criterion] = []
        self.weights: List[float] = []

    def add_criterion(self, citerion: Criterion, weight: float):
        self.criteria.append(citerion)
        self.weights.append(weight)

    def calculate_reward(self, simulation_output):
        partial_rewards = []
        for criterion in self.criteria:
            partial_rewards.append(criterion.calculate_reward(simulation_output))

        print([round(x,3) for x in  partial_rewards])
        total_reward = -sum([a * b for a, b in zip(partial_rewards, self.weights)])
        return round(total_reward,3)
