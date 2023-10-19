import json
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
import os
from typing import Any, Type

from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import direct, dual_annealing
from rostok.control_chrono.controller import RobotControllerChrono

from rostok.control_chrono.tendon_controller import TendonController_2p, TendonControllerParameters
from rostok.criterion.criterion_calculation import SimulationReward
from rostok.criterion.simulation_flags import EventFlyingApart
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import (
    get_joint_matrix_from_graph, get_joint_vector_from_graph)
from rostok.simulation_chrono.simulation_scenario import GraspScenario, ParametrizedSimulation
from rostok.trajectory_optimizer.trajectory_generator import (
    joint_root_paths)
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
        return False

    
    def bound_parameters(self, graph: GraphGrammar, bounds: tuple):
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
        start_pos = self.build_starting_positions(graph) # pylint: disable=assignment-from-none
        is_vis = self.is_vis_decision(graph)
        simout = sim.run_simulation(graph, control_data, start_pos, is_vis)
        rew = self.rewarder.calculate_reward(simout)
        return rew, x, sim




class MockOptiVar(BasePrepareOptiVar):

    def reward_one_sim_scenario(self, x: list,
                                graph: GraphGrammar,
                                sim: ParametrizedSimulation):

        return np.mean(x), x, sim


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

class BruteForceOptimisation1D():
    def __init__(self,
                 variants: list,
                 simulation_scenario: list[ParametrizedSimulation],
                 prepare_reward: BasePrepareOptiVar,
                 num_cpu_workers=1,
                 chunksize=1,
                 timeout_parallel = 60*5):
        self.variants = variants
        self.simulation_scenario = simulation_scenario
        self.prepare_reward = prepare_reward
        self.num_cpu_workers = num_cpu_workers
        self.chunksize = chunksize
        self.timeout_parallel = timeout_parallel

    def parallel_calculate_reward(self, graph: GraphGrammar):
        if not is_valid_graph(graph):
            return (0.01, [])

        all_variants_control = list(product(self.variants, repeat=len(joint_root_paths(graph))))
        if isinstance(self.simulation_scenario, list):
            all_simulations = list(product(all_variants_control, self.simulation_scenario))
            input_dates = [(np.array(put[0]), graph, put[1]) for put in all_simulations]
        else:
            all_simulations = list(product(all_variants_control, [self.simulation_scenario]))
            input_dates = [(np.array(put[0]), graph, put[1]) for put in all_simulations]
        np.random.shuffle(input_dates)
        parallel_results = []
        if self.num_cpu_workers > 1:
            cpus = len(input_dates) + 1 if len(input_dates) < self.num_cpu_workers else self.num_cpu_workers
            print(f"Use CPUs processor: {cpus}, input dates: {len(input_dates)}")
            
            try:
                parallel_results = Parallel(
                    cpus,
                    backend="multiprocessing",
                    verbose=100,
                    timeout=self.timeout_parallel,
                    batch_size=self.chunksize)(
                        delayed(self.prepare_reward.reward_one_sim_scenario)(i[0], i[1], i[2]) for i in input_dates)
            except TimeoutError:
                print("TIMEOUT")
                return (0.01, [])
            
        else:
            for i in input_dates:
                res = self.prepare_reward.reward_one_sim_scenario(i[0], i[1], i[2])
                parallel_results.append(res)
        
        result_group_object = {sim_scen.get_scenario_name(): [] for sim_scen in self.simulation_scenario}
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
        print('n_joints:', n_joints)
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
        parameters = parameters.round(3)
        data = {"initial_value": parameters}

        return data

    def _postprocessing_parameters(self, parameters):

        return np.round(parameters, 3)

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

    def simulate_with_control_parameters(self, controller_data, graph, simulation_scenario):
        starting_positions = self.build_starting_positions(graph)
        return simulation_scenario.run_simulation(graph,
                                                  controller_data,
                                                  starting_positions,
                                                  vis=False,
                                                  delay=False)

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
        joint_paths = joint_root_paths(graph)
        n_branches = len(joint_paths)
        print('n_branches:', n_branches)
        if n_branches == 0 or n_branches > 4:
            return []
        else:
            lengths = [len(x) for x in joint_paths]
            if max(lengths) > 4:
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
        if list(
                filter(lambda x: isinstance(x, EventFlyingApart),
                       simulator_scenario.event_container))[0].state:
            return 0.03
        reward = self.rewarder.calculate_reward(sim_output)
        return -reward

    def _transform_parameters2data(self, parameters, *args):
        """Method define transofrm algorigm parameters to data control

        Args:
            parameters (list): Parameter list for optimizing

        Returns:
            dict: Dictionary of data control
        """
        parameters = parameters.round(self.round_const)
        self.data.forces = list(parameters)
        data = deepcopy(self.data)
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
class Result:
    fun: float = 0
    x: list[float] = field(default_factory=list)


class TendonOptimizerCombinationForce(TendonOptimizer):

    def __init__(self,
                 simulation_scenario,
                 rewarder: SimulationReward,
                 data: TendonControllerParameters,
                 tendon_forces: list[float],
                 starting_finger_angles=45,
                 num_cpu_workers=1,
                 chunksize=1,
                 timeout_parallel = 60*5):
        """Brute force optimization of tendon forces for controlling the mechanism. In subclass, it have to override method: bound_parameter, _transform_parameter2data and run_optimization. Number of cpu workers define number of parallel processes.

        Args:
            simulation_scenario (_type_): Scenario of simulation for virtual experiment
            rewarder (SimulationReward): Instance of the class on which the objective function will be calculated
            data (TendonControllerParameters): Parameters of control class
            tendon_forces (list[float]): List of tendon force for brute force optimization.
            starting_finger_angles (int, optional): Initial angle of fingers. Defaults to 45.
            num_cpu_workers (int, optional): Number of parallel process. When set to "auto", the algorithm selects the number of workers by itself. Defaults to 1.
            chunksize (int, optional): Number of batch for one cpu worker. When set to "auto", the algorithm selects the number of workers by itself. Defaults to 1.
        """
        mock_optimization_bounds = (0, 15)
        mock_optimization_limit = 10
        self.tendon_forces = tendon_forces
        super().__init__(simulation_scenario, rewarder, data, starting_finger_angles,
                         mock_optimization_bounds, mock_optimization_limit)
        self.num_cpu_workers = num_cpu_workers
        self.chunksize = chunksize
        self.timeout_parallel = timeout_parallel

        if self.num_cpu_workers == "auto":
            self.num_cpu_workers = os.cpu_count() - 2

    def run_optimization(self, callback, multi_bound, args):
        graph = args[0]
        number_of_fingers = len(joint_root_paths(graph))
        all_variants_control = list(product(self.tendon_forces, repeat=number_of_fingers))
        results = []
        for variant in all_variants_control:
            res = callback(np.array(variant), *args)
            res_comp = Result(res, np.array(variant))
            results.append(res_comp)
        result = min(results, key=lambda i: i.fun)
        return result

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
        # print(f"Data: correct!, {data}")
        sim_output = self.simulate_with_control_parameters(data, graph, simulator_scenario)
        if list(
                filter(lambda x: isinstance(x, EventFlyingApart),
                       simulator_scenario.event_container))[0].state:
            return parameters, simulator_scenario, 0.03
        reward = self.rewarder.calculate_reward(sim_output)
        return parameters, simulator_scenario, reward

    def __parallel_calculate_reward(self, graph: GraphGrammar):
        multi_bound = self.bound_parameters(graph)

        if not multi_bound:
            return (0, [])

        all_variants_control = list(product(self.tendon_forces, repeat=len(joint_root_paths(graph))))
        if isinstance(self.simulation_scenario, list):
            object_weight = {sim_scen[0].grasp_object_callback: sim_scen[1] for sim_scen in self.simulation_scenario}
            all_simulations = list(product(all_variants_control, self.simulation_scenario))
            input_dates = [(np.array(put[0]), graph, put[1][0]) for put in all_simulations]
        else:
            object_weight = {self.simulation_scenario.grasp_object_callback: 1}
            all_simulations = list(product(all_variants_control, [self.simulation_scenario]))
            input_dates = [(np.array(put[0]), graph, put[1]) for put in all_simulations]
        np.random.shuffle(input_dates)

        cpus = len(input_dates) + 1 if len(input_dates) < self.num_cpu_workers else self.num_cpu_workers
        print(f"Use CPUs processor: {cpus}, input dates: {len(input_dates)}")
        parallel_results = []
        try:
            parallel_results = Parallel(
                cpus,
                backend="multiprocessing",
                verbose=100,
                timeout=self.timeout_parallel,
                batch_size=self.chunksize)(
                    delayed(self._parallel_reward_with_parameters)(i) for i in input_dates)
        except TimeoutError:
            print("TIMEOUT")
            return (0.01, [])
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

    def calculate_reward(self, graph: GraphGrammar):
        if self.num_cpu_workers > 1:
            return self.__parallel_calculate_reward(graph)
        return super().calculate_reward(graph)