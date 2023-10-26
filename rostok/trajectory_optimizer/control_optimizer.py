from functools import partial
import json
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
import os
from typing import Any, Type
from collections.abc import Iterable
from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import direct, dual_annealing
from rostok.control_chrono.controller import ConstController, RobotControllerChrono

from rostok.control_chrono.tendon_controller import TendonController_2p, TendonControllerParameters
from rostok.criterion.criterion_calculation import SimulationReward
from rostok.criterion.simulation_flags import EventFlyingApart
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import (get_joint_matrix_from_graph,
                                                    get_joint_vector_from_graph)
from rostok.simulation_chrono.simulation_scenario import GraspScenario, ParametrizedSimulation
from rostok.trajectory_optimizer.trajectory_generator import (joint_root_paths)
from rostok.utils.json_encoder import RostokJSONEncoder


def is_valid_graph(graph: GraphGrammar):
    n_joints = len(get_joint_vector_from_graph(graph))
    return n_joints > 0


def build_equal_starting_positions(graph: GraphGrammar, starting_finger_angles):
    joint_matrix = get_joint_matrix_from_graph(graph)
    for i in range(len(joint_matrix)):
        for j in range(len(joint_matrix[i])):
            if j == 0:
                joint_matrix[i][j] = starting_finger_angles
            else:
                joint_matrix[i][j] = 0
    return joint_matrix


def build_control_graph_from_joint(graph: GraphGrammar, joint_dict: dict):
    joints = get_joint_vector_from_graph(graph)
    control_sequence = []
    for idx in joints:
        node = graph.get_node_by_id(idx)
        control_sequence.append(joint_dict[node])
    return control_sequence


class GraphRewardCalculator:

    def __init__(self):
        pass

    @abstractmethod
    def calculate_reward(self, graph: GraphGrammar):
        pass

    def print_log(self):
        pass

    def __repr__(self) -> str:
        json_data = json.dumps(self, cls=RostokJSONEncoder)
        return json_data

    def __str__(self) -> str:
        json_data = json.dumps(self, indent=4, cls=RostokJSONEncoder)
        return json_data


class BasePrepareOptiVar():

    def __init__(self,
                 each_control_params: Any,
                 control_class: Type[RobotControllerChrono],
                 rewarder: SimulationReward,
                 params_start_pos=None):
        self.each_control_params = each_control_params
        self.control_class = control_class
        self.rewarder = rewarder
        self.is_vis = False
        self.params_start_pos = params_start_pos

    @abstractmethod
    def x_to_control_params(self, graph: GraphGrammar, x: list):
        pass

    def build_starting_positions(self, graph: GraphGrammar):
        return None

    def is_vis_decision(self, graph: GraphGrammar):
        return True

    def bound_parameters(self, graph: GraphGrammar, bounds: tuple[float, float]):
        """A method for determining the relationship between boundaries and mechanism.
        Default implementation
        Args:
            graph (GraphGrammar): _description_

        Returns:
            _type_: _description_
        """
        n_joints = len(get_joint_vector_from_graph(graph))
        print('n_joints:', n_joints)
        multi_bound = []
        for _ in range(n_joints):
            multi_bound.append(bounds)

        return multi_bound

    def reward_one_sim_scenario(self, x: list, graph: GraphGrammar, sim: ParametrizedSimulation):
        control_data = self.x_to_control_params(graph, x)
        start_pos = self.build_starting_positions(graph)  # pylint: disable=assignment-from-none
        is_vis = self.is_vis_decision(graph) and self.is_vis
        simout = sim.run_simulation(graph, control_data, start_pos, is_vis)
        rew = self.rewarder.calculate_reward(simout)
        print('x:', x)
        return rew, x, sim

class TendonForceOptiVar(BasePrepareOptiVar):

    def __init__(self,
                 each_control_params: TendonControllerParameters,
                 rewarder: SimulationReward,
                 params_start_pos=None):
        super().__init__(each_control_params, TendonController_2p, rewarder, params_start_pos)

    def x_to_control_params(self, graph: GraphGrammar, x: list):
        """Method define transofrm algorigm parameters to data control

        Args:
            parameters (list): Parameter list for optimizing

        Returns:
            dict: Dictionary of data control
        """
        np_array_x = np.array(x)
        parameters = np_array_x.round(3)
        self.each_control_params.forces = list(parameters)
        data = deepcopy(self.each_control_params)
        return data

    def build_starting_positions(self, graph: GraphGrammar):
        if self.params_start_pos:
            return build_equal_starting_positions(graph, self.params_start_pos)
        return None

    def bound_parameters(self, graph: GraphGrammar, bounds: tuple[float, float]):
        """A method for determining the relationship between boundaries and mechanism.
        Default implementation
        Args:
            graph (GraphGrammar): _description_

        Returns:
            _type_: _description_
        """
        n_joints = len(joint_root_paths(graph))
        print('n_joints:', n_joints)
        multi_bound = []
        for _ in range(n_joints):
            multi_bound.append(bounds)

        return multi_bound


class BruteForceOptimisation1D(GraphRewardCalculator):

    def __init__(self,
                 variants: list,
                 simulation_scenario: list[ParametrizedSimulation],
                 prepare_reward: BasePrepareOptiVar,
                 num_cpu_workers=1,
                 chunksize=1,
                 timeout_parallel=60 * 5):
        self.variants = variants
        self.simulation_scenario = simulation_scenario
        self.prepare_reward = prepare_reward
        self.num_cpu_workers = num_cpu_workers
        self.chunksize = chunksize
        self.timeout_parallel = timeout_parallel

    def generate_all_combine(self, graph: GraphGrammar):
        number_control_varibales = len(self.prepare_reward.bound_parameters(graph, (0, 1)))
        all_variants_control = list(product(self.variants, repeat=number_control_varibales))
        return all_variants_control

    def calculate_reward(self, graph: GraphGrammar):
        if not is_valid_graph(graph):
            return (0.01, [])

        all_variants_control = self.generate_all_combine(graph)
        if isinstance(self.simulation_scenario, list):
            all_simulations = list(product(all_variants_control, self.simulation_scenario))
            input_dates = [(np.array(put[0]), graph, put[1]) for put in all_simulations]
        else:
            all_simulations = list(product(all_variants_control, [self.simulation_scenario]))
            input_dates = [(np.array(put[0]), graph, put[1]) for put in all_simulations]
        np.random.shuffle(input_dates)
        parallel_results = []
        if self.num_cpu_workers > 1:
            cpus = len(input_dates) + 1 if len(
                input_dates) < self.num_cpu_workers else self.num_cpu_workers
            print(f"Use CPUs processor: {cpus}, input dates: {len(input_dates)}")

            try:
                parallel_results = Parallel(
                    cpus,
                    backend="multiprocessing",
                    verbose=100,
                    timeout=self.timeout_parallel,
                    batch_size=self.chunksize)(
                        delayed(self.prepare_reward.reward_one_sim_scenario)(i[0], i[1], i[2])
                        for i in input_dates)
            except TimeoutError:
                print("TIMEOUT")
                return (0.01, [])

        else:
            for i in input_dates:
                res = self.prepare_reward.reward_one_sim_scenario(i[0], i[1], i[2])
                parallel_results.append(res)

        result_group_object = {
            sim_scen.get_scenario_name(): [] for sim_scen in self.simulation_scenario
        }
        for results in parallel_results:
            scen_name = results[2].get_scenario_name()
            result_group_object[scen_name].append((results[1], results[0]))

        reward = 0
        control = []
        for value in result_group_object.values():
            best_res = max(value, key=lambda i: i[1])
            reward += best_res[1]
            control.append(best_res[0])

        return (reward, control)


class GlobalOptimisationEachSim(GraphRewardCalculator):

    def __init__(self,
                 simulation_scenario: list[ParametrizedSimulation],
                 prepare_reward: BasePrepareOptiVar,
                 bound: tuple[float, float],
                 args_for_optimiser=None,
                 optimisation_tool=direct):
        self.optimisation_tool = optimisation_tool
        self.simulation_scenario = simulation_scenario
        self.prepare_reward = prepare_reward
        self.args_for_optimiser = args_for_optimiser
        self.bound = bound

    def calculate_reward(self, graph: GraphGrammar):
        self.prepare_reward.bound_parameters(graph, (0, 1))
        resaults = []
        for sim in self.simulation_scenario:
            x_input_function = partial(self.prepare_reward.reward_one_sim_scenario,
                                       graph=graph,
                                       sim=sim)
            x_input_function_first_arg = lambda x: -x_input_function(x)[0]
            res = self.optimisation_tool(x_input_function_first_arg,
                                         bounds=self.prepare_reward.bound_parameters(
                                             graph, self.bound),
                                         **self.args_for_optimiser)
            resaults.append(res)
        controls = []
        rew = 0
        for res_i in resaults:
            rew += res_i.fun
            controls.append(res_i.x)
        return (-rew, controls)


class ConstTorqueOptiVar(BasePrepareOptiVar):

    def __init__(self, rewarder: SimulationReward, params_start_pos=None):
        super().__init__([], ConstController, rewarder, params_start_pos)

    def x_to_control_params(self, graph: GraphGrammar, x: list):
        """Method define transofrm algorigm parameters to data control

        Args:
            parameters (list): Parameter list for optimizing

        Returns:
            dict: Dictionary of data control
        """
        np_array_x = np.array(x)
        parameters = np_array_x.round(3)
        data = {"initial_value": parameters}
        return data

    def build_starting_positions(self, graph: GraphGrammar):
        if self.params_start_pos:
            return build_equal_starting_positions(graph, self.params_start_pos)
        return None

    def bound_parameters(self, graph: GraphGrammar, bounds: tuple[float, float]):
        """A method for determining the relationship between boundaries and mechanism 

        Args:
            graph (GraphGrammar): _description_

        Returns:
            _type_: _description_
        """
        n_joints = len(get_joint_vector_from_graph(graph))
        print('n_joints:', n_joints)
        multi_bound = []
        for _ in range(n_joints):
            multi_bound.append(bounds)

        return multi_bound


class FromGraphOptimizer(GraphRewardCalculator):

    def __init__(
        self,
        params_dict: dict,
        simulation_scenario: list[ParametrizedSimulation],
        prepare_reward: BasePrepareOptiVar,
    ):
        self.params_dict = params_dict
        self.simulation_scenario = simulation_scenario
        self.prepare_reward = prepare_reward

    def create_vector_from_graph(self, graph: GraphGrammar):
        control_vec = build_control_graph_from_joint(graph, self.params_dict)
        return control_vec

    def calculate_reward(self, graph: GraphGrammar):
        control_vec = self.create_vector_from_graph(graph)
        if isinstance(self.simulation_scenario, Iterable):
            all_reward = 0
            for simulation in self.simulation_scenario:
                single_reward = self.prepare_reward.reward_one_sim_scenario(
                    control_vec, graph, simulation)
                all_reward += single_reward
            return (all_reward, control_vec)
        else:
            rew = self.prepare_reward.reward_one_sim_scenario(control_vec, graph,
                                                              self.simulation_scenario)
            return (rew, control_vec)