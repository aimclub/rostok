from multiprocessing import Pool, TimeoutError
import os

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
import time
import numpy as np
import json
import types
from rostok.control_chrono.tendon_controller import TendonControllerParameters
from rostok.graph_grammar.node_block_typing import get_joint_matrix_from_graph
from scipy.optimize import direct, dual_annealing, shgo

from rostok.criterion.criterion_calculation import SimulationReward
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from enum import Enum
from rostok.simulation_chrono.simulation_scenario import ParametrizedSimulation
from rostok.trajectory_optimizer.trajectory_generator import cable_length_linear_control, linear_control, joint_root_paths, tendon_like_control
from rostok.utils.json_encoder import RostokJSONEncoder
from itertools import product

class GraphRewardCalculator:

    def __init__(self):
        pass

    @abstractmethod
    def calculate_reward(self, graph: GraphGrammar):
        pass

    @abstractmethod
    def print_log(self):
        pass

    def __repr__(self) -> str:
        json_data = json.dumps(self, cls=RostokJSONEncoder)
        return json_data

    def __str__(self) -> str:
        json_data = json.dumps(self, indent=4, cls=RostokJSONEncoder)
        return json_data


class CalculatorWithConstTorqueOptimization(GraphRewardCalculator):

    def __init__(self,
                 simulation_scenario,
                 rewarder: SimulationReward,
                 optimization_bounds=(0, 15),
                 optimization_limit=10):
        """Base class optimizing constant torque for controlling the mechanism. In subclass, it have to override method: bound_parameter, _transform_parameter2data and run_optimization.

        Args:
            simulation_scenario (Union[list[tuple[ParametrizedSimulation, int]], ParametrizedSimulation]): Define simulation scenario for virtual experiment and weights for each.
            rewarder (SimulationReward): Instance of the class on which the objective function will be calculated
            optimization_bounds (tuple, optional): Args define the boundaries of the variables to be optimized. Defaults to (0, 15).
            optimization_limit (int, optional): The maximum number of optimization iterations. Defaults to 10.
        """
        self.simulation_scenario = simulation_scenario
        self.rewarder: SimulationReward = rewarder
        self.bounds = optimization_bounds
        self.limit = optimization_limit

    def simulate_with_control_parameters(self, data, graph):
        return self.simulation_scenario.run_simulation(graph, data)

    def calculate_reward(self, graph: GraphGrammar):
        """Constant moment optimization method using scenario simulation and rewarder for calculating objective function.

        Args:
            graph (GraphGrammar): A graph of the mechanism for which the control is to be found

        Returns:
            (float, np.ndarray): Return the reward and optimized variables of the best candidate
        """
        multi_bound = self.bound_parameters(graph)

        if not multi_bound:
            return (0.01, [])
        if isinstance(self.simulation_scenario, list):
            reward = 0.01
            optim_parameters = np.array([])
            for sim_scene in self.simulation_scenario:
                result = self.run_optimization(self._reward_with_parameters,
                                               multi_bound,
                                               args=(graph, sim_scene[0]))

                reward -= result.fun * sim_scene[1]
                processed_parameters = self._postprocessing_parameters(result.x)
                if optim_parameters.size == 0:
                    optim_parameters = processed_parameters
                else:
                    optim_parameters = np.vstack((optim_parameters, processed_parameters))

        else:
            result = self.run_optimization(self._reward_with_parameters,
                                           multi_bound,
                                           args=(graph, self.simulation_scenario))

            reward = -result.fun
            optim_parameters = self._postprocessing_parameters(result.x)

        return (reward, optim_parameters)

    def optim_parameters2data_control(self, parameters, *args):
        """Method convert optimizing variables to structure for class of control

        Args:
            parameters (list): List of parameters to be optimized. For several simulation scenarios this will be a 2d-list

        Returns:
            dict: Dictionary defining the parameters of the control class
        """
        parameters = np.array(parameters)
        if isinstance(self.simulation_scenario, list):
            list_args = [args for __ in range(len(parameters))]
            data_control = list(map(self._transform_parameters2data, parameters, list_args))
        else:
            data_control = self._transform_parameters2data(parameters, args)
        return data_control

    def bound_parameters(self, graph: GraphGrammar):
        """A method for determining the relationship between boundaries and mechanism 

        Args:
            graph (GraphGrammar): _description_

        Returns:
            _type_: _description_
        """
        n_joints = len(get_joint_vector_from_graph(graph))
        # print('n_joints:', n_joints)
        multi_bound = []
        for _ in range(n_joints):
            multi_bound.append(self.bounds)

        return multi_bound

    def _reward_with_parameters(self, parameters, graph, simulator_scenario):
        """Objective function to be optimized

        Args:
            parameters (np.ndarray): Array variables of objective function
            graph (GraphGrammar): Graph of mechanism for which the optimization do
            simulator_scenario (ParamtrizedAimulation): Simulation scenario in which data is collected for calcule the objective function

        Returns:
            float: Value of objective function
        """
        data = self._transform_parameters2data(parameters)
        sim_output = simulator_scenario.run_simulation(graph, data)
        reward = self.rewarder.calculate_reward(sim_output)
        return -reward

    def _transform_parameters2data(self, parameters, *args):
        """Method define transofrm algorigm parameters to data control

        Args:
            parameters (list): Parameter list for optimizing

        Returns:
            dict: Dictionary of data control
        """
        parameters = parameters.round(6)
        data = {"initial_value": parameters}

        return data

    def _postprocessing_parameters(self, parameters):

        return np.round(parameters, 6)

    @abstractmethod
    def run_optimization(self, callback, multi_bound, args):
        pass


class CalculatorWithOptimizationDirect(CalculatorWithConstTorqueOptimization):

    def run_optimization(self, callback, multi_bound, args):
        result = direct(callback, multi_bound, maxiter=self.limit, args=args)
        return result


class CalculatorWithOptimizationDualAnnealing(CalculatorWithConstTorqueOptimization):

    def run_optimization(self, callback, multi_bound, args):
        result = dual_annealing(callback, multi_bound, maxiter=self.limit, args=args)
        return result


class CalculatorWithGraphOptimization(GraphRewardCalculator):

    def __init__(self, simulation_scenario, rewarder: SimulationReward, torque_dict):
        self.simulation_scenario = simulation_scenario
        self.rewarder: SimulationReward = rewarder
        self.torque_dict = torque_dict

    def build_control_from_graph(self, graph: GraphGrammar):
        joints = get_joint_vector_from_graph(graph)
        control_sequence = []
        for idx in joints:
            node = graph.get_node_by_id(idx)
            control_sequence.append(self.torque_dict[node])
        return control_sequence

    def calculate_reward(self, graph: GraphGrammar):

        n_joints = get_joint_vector_from_graph(graph)
        if n_joints == 0:
            return (0, [])
        control_sequence = self.build_control_from_graph(graph)
        data = {"initial_value": control_sequence}
        simulation_output = self.simulation_scenario.run_simulation(graph, data)
        reward = self.rewarder.calculate_reward(simulation_output)
        return (reward, control_sequence)


class OptimizationParametr(Enum):
    """Enum for select optimization value.
    Start means constant part of linear function, multiplier 
    means slope of line.
    """
    START = 1
    MULTIPLIER = 2


class ConstTorqueOptimizationBranchTemplate(CalculatorWithConstTorqueOptimization):
    """A template class for constant torque optimization on branches of a graph.
    For use you need implement run_optimization and generate_control_value_on_branch.
    run_optimization for select optimizer.
    generate_control_value_on_branch for generate control.

    Args:
        GraphRewardCalculator: _description_
    """

    def __init__(self,
                 simulation_scenario,
                 rewarder: SimulationReward,
                 optimization_bounds=(0, 15),
                 optimization_limit=10,
                 select_optimisation_value=OptimizationParametr.START,
                 const_parameter=-0.5):
        super().__init__(simulation_scenario, rewarder, optimization_bounds, optimization_limit)
        self.select_optimisation_value = select_optimisation_value
        self.const_parameter = const_parameter

    def extend_parameters_by_const(self, parameters: list[float]) -> list[tuple[float, float]]:
        """"Extends the control parameters list by adding the constant parameter

        Args:
            parameters (list[float]): 

        Raises:
            Exception: _description_

        Returns:
            list[tuple[float, float]]: 
        """

        select_index_order = {
            OptimizationParametr.START: [0, 1],
            OptimizationParametr.MULTIPLIER: [1, 0]
        }.get(self.select_optimisation_value)

        if select_index_order is None:
            raise Exception("Wrong select_optimisation_value")
        parameters_2d = []
        for param in parameters:
            buf = [0, 0]
            buf[select_index_order[0]] = param
            buf[select_index_order[1]] = self.const_parameter
            parameters_2d.append(tuple(buf))
        return parameters_2d

    def bound_parameters(self, graph: GraphGrammar):
        n_branches = len(joint_root_paths(graph))
        print('n_branches:', n_branches)
        if n_branches == 0:
            return []
        multi_bound = []
        for _ in range(n_branches):
            multi_bound.append(self.bounds)

        return multi_bound

    def _reward_with_parameters(self, parameters, graph, simulator_scenario):
        data = self._transform_parameters2data(parameters, graph)
        sim_output = simulator_scenario.run_simulation(graph, data)
        reward = self.rewarder.calculate_reward(sim_output)
        return -reward

    def _transform_parameters2data(self, parameters, graph):
        parameters = parameters.round(6)
        if isinstance(graph, tuple):
            graph = graph[0]

        parameters = list(parameters)
        parameters_2d = self.extend_parameters_by_const(parameters)
        data = self.generate_control_value_on_branch(graph, parameters_2d)

        return data

    @abstractmethod
    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        pass


class ConstControlOptimizationDirect(ConstTorqueOptimizationBranchTemplate, ABC):
    """A template class for constant torque optimization on branches of a graph
    with direct optimization.  

    Args:
        ConstTorqueOptimizationBranchTemplate (_type_): _description_
    """

    def run_optimization(self, callback, multi_bound, args):
        result = direct(callback, multi_bound, maxiter=self.limit, args=args)
        return result


class LinearControlOptimizationDirect(ConstControlOptimizationDirect):

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return linear_control(graph, parameters_2d)


class TendonLikeControlOptimization(ConstControlOptimizationDirect):

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return tendon_like_control(graph, parameters_2d)


class LinearCableControlOptimization(ConstControlOptimizationDirect):

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return cable_length_linear_control(graph, parameters_2d)


class TendonOptimizer(GraphRewardCalculator):

    def __init__(self,
                 simulation_scenario,
                 rewarder: SimulationReward,
                 data: TendonControllerParameters,
                 starting_finger_angles=45,
                 optimization_bounds=(0, 15),
                 optimization_limit=10):
        self.data: TendonControllerParameters = data
        self.simulation_scenario = simulation_scenario
        self.rewarder: SimulationReward = rewarder
        self.bounds = optimization_bounds
        self.limit = optimization_limit
        self.round_const = 4
        self.starting_finger_angles = starting_finger_angles

    def build_starting_positions(self, graph: GraphGrammar):
        if self.starting_finger_angles:
            joint_matrix = get_joint_matrix_from_graph(graph)
            for i in range(len(joint_matrix)):
                for j in range(len(joint_matrix[i])):
                    if j == 0:
                        joint_matrix[i][j] = self.starting_finger_angles
                    else:
                        joint_matrix[i][j] = 0
            return joint_matrix
        else:
            return []


    def simulate_with_control_parameters(self, data, graph, simulation_scenario):
        starting_positions = self.build_starting_positions(graph)
        return simulation_scenario.run_simulation(graph, data, starting_positions, vis=False, delay=False)

    def calculate_reward(self, graph: GraphGrammar):
        """Constant moment optimization method using scenario simulation and rewarder for calculating objective function.

        Args:
            graph (GraphGrammar): A graph of the mechanism for which the control is to be found

        Returns:
            (float, np.ndarray): Return the reward and optimized variables of the best candidate
        """
        multi_bound = self.bound_parameters(graph)

        if not multi_bound:
            return (0, [])

        if isinstance(self.simulation_scenario, list):
            reward = 0
            optim_parameters = np.array([])
            for sim_scene in self.simulation_scenario:
                result = self.run_optimization(self._reward_with_parameters,
                                               multi_bound,
                                               args=(graph, sim_scene[0]))

                reward -= result.fun * sim_scene[1]
                processed_parameters = self._postprocessing_parameters(result.x)
                if optim_parameters.size == 0:
                    optim_parameters = processed_parameters
                else:
                    optim_parameters = np.vstack((optim_parameters, processed_parameters))

        else:
            result = self.run_optimization(self._reward_with_parameters,
                                           multi_bound,
                                           args=(graph, self.simulation_scenario))

            reward = -result.fun
            optim_parameters = self._postprocessing_parameters(result.x)

        return (reward, optim_parameters)

    def optim_parameters2data_control(self, parameters, *args):
        """Method convert optimizing variables to structure for class of control

        Args:
            parameters (list): List of parameters to be optimized. For several simulation scenarios this will be a 2d-list

        Returns:
            dict: Dictionary defining the parameters of the control class
        """
        parameters = np.array(parameters)
        if isinstance(self.simulation_scenario, list):
            list_args = [args for __ in range(len(parameters))]
            data_control = list(map(self._transform_parameters2data, parameters, list_args))
        else:
            data_control = self._transform_parameters2data(parameters, args)
        return data_control

    def bound_parameters(self, graph: GraphGrammar):
        n_branches = len(joint_root_paths(graph))
        print('n_branches:', n_branches)
        if n_branches == 0 or n_branches > 4:
            return []
        multi_bound = []
        for _ in range(n_branches):
            multi_bound.append(self.bounds)

        return multi_bound

    def _reward_with_parameters(self, parameters, graph, simulator_scenario):
        """Objective function to be optimized

        Args:
            parameters (np.ndarray): Array variables of objective function
            graph (GraphGrammar): Graph of mechanism for which the optimization do
            simulator_scenario (ParamtrizedAimulation): Simulation scenario in which data is collected for calcule the objective function

        Returns:
            float: Value of objective function
        """
        data = self._transform_parameters2data(parameters)
        sim_output = self.simulate_with_control_parameters(data, graph, simulator_scenario)
        reward = self.rewarder.calculate_reward(sim_output)
        return -reward

    def _parallel_reward_with_parameters(self, input):
        """Objective function to be optimized

        Args:
            parameters (np.ndarray): Array variables of objective function
            graph (GraphGrammar): Graph of mechanism for which the optimization do
            simulator_scenario (ParamtrizedAimulation): Simulation scenario in which data is collected for calcule the objective function

        Returns:
            float: Value of objective function
        """
        parameters, graph, simulator_scenario = input
        data = self._transform_parameters2data(parameters)
        sim_output = self.simulate_with_control_parameters(data, graph, simulator_scenario)
        reward = self.rewarder.calculate_reward(sim_output)
        return parameters, simulator_scenario, reward

    def _transform_parameters2data(self, parameters, *args):
        """Method define transofrm algorigm parameters to data control

        Args:
            parameters (list): Parameter list for optimizing

        Returns:
            dict: Dictionary of data control
        """
        parameters = parameters.round(self.round_const)
        self.data.forces = list(parameters)
        data = self.data
        return data

    def _postprocessing_parameters(self, parameters):

        return np.round(parameters, self.round_const)

    @abstractmethod
    def run_optimization(self, callback, multi_bound, args):
        pass


class TendonOptimizerDirect(TendonOptimizer):
    """A template class for constant torque optimization on branches of a graph
    with direct optimization.  

    Args:
        ConstTorqueOptimizationBranchTemplate (_type_): _description_
    """

    def run_optimization(self, callback, multi_bound, args):
        result = direct(callback, multi_bound, maxiter=self.limit, args=args)
        return result

@dataclass
class Resault:
    fun: float = 0
    x : list[float] = field(default_factory=list)

class TendonOptimizerCombinationForce(TendonOptimizer):

    def __init__(self,
                 simulation_scenario,
                 rewarder: SimulationReward,
                 data: TendonControllerParameters,
                 tendon_forces: list[float],
                 starting_finger_angles=45):
        mock_optimization_bounds = (0, 15)
        mock_optimization_limit = 10
        self.tendon_forces = tendon_forces
        super().__init__(simulation_scenario, rewarder, data, starting_finger_angles,
                         mock_optimization_bounds, mock_optimization_limit)


    def run_optimization(self, callback, multi_bound, args):
        graph = args[0]
        number_of_fingers = len(joint_root_paths(graph))
        all_variants_control = list(
            product(self.tendon_forces, repeat=number_of_fingers))
        results = []
        for variant in all_variants_control:
            res = callback(np.array(variant), *args)
            res_comp = Resault(res, np.array(variant))
            results.append(res_comp)
        result = min(results, key=lambda i: i.fun)
        return result
    
class ParralelOptimizerCombinationForce(TendonOptimizer):
    def __init__(self,
                 simulation_scenario,
                 rewarder: SimulationReward,
                 data: TendonControllerParameters,
                 tendon_forces: list[float],
                 starting_finger_angles=45):
        mock_optimization_bounds = (0, 15)
        mock_optimization_limit = 10
        self.tendon_forces = tendon_forces
        super().__init__(simulation_scenario, rewarder, data, starting_finger_angles,
                         mock_optimization_bounds, mock_optimization_limit)

    def calculate_reward(self, graph: GraphGrammar):
        """Constant moment optimization method using scenario simulation and rewarder for calculating objective function.

        Args:
            graph (GraphGrammar): A graph of the mechanism for which the control is to be found

        Returns:
            (float, np.ndarray): Return the reward and optimized variables of the best candidate
        """
        multi_bound = self.bound_parameters(graph)

        if not multi_bound:
            return (0, [])
        
        cpus = os.cpu_count() - 2
        print(f"CPUs processor: {cpus}")
        all_variants_control = list(product(self.tendon_forces, repeat=len(joint_root_paths(graph))))
        object_weight = {sim_scen[0].grasp_object_callback: sim_scen[1] for sim_scen in self.simulation_scenario}
        all_simulations = list(product(all_variants_control, self.simulation_scenario))
        input_dates = [(np.array(put[0]), graph, put[1][0]) for put in all_simulations]
        np.random.shuffle(input_dates)
        
        cpus = len(input_dates) + 1 if len(input_dates) < cpus else cpus
        print(f"Use CPUs processor: {cpus}")
        parallel_results = []
        with Pool(processes=cpus) as pool:
            for out in pool.imap_unordered(self._parallel_reward_with_parameters, input_dates):
                parallel_results.append(out)

        result_group_object = {sim_scen[0].grasp_object_callback: [] for sim_scen in self.simulation_scenario}
        for results in parallel_results:
            obj = results[1].grasp_object_callback
            result_group_object[obj].append((results[0], results[2]*object_weight[obj]))
        
        reward = 0
        control = []
        for value in result_group_object.values():
            best_res = max(value, key=lambda i: i[1])
            reward += best_res[1]
            control.append(best_res[0])
        
        return (reward, control)
    
    def run_optimization(self, callback, multi_bound, args):
        graph = args[0]
        number_of_fingers = len(joint_root_paths(graph))
        all_variants_control = list(
            product(self.tendon_forces, repeat=number_of_fingers))
        results = []
        for variant in all_variants_control:
            res = callback(np.array(variant), *args)
            res_comp = Resault(res, np.array(variant))
            results.append(res_comp)
        result = min(results, key=lambda i: i.fun)
        

        return result
