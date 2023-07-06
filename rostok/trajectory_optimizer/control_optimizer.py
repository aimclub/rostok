from abc import abstractmethod

from scipy.optimize import direct, dual_annealing, shgo

from rostok.criterion.criterion_calculation import SimulationReward
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from enum import Enum
from rostok.trajectory_optimizer.trajectory_generator import linear_control, joint_root_paths, tendon_like_control


class GraphRewardCalculator:

    def __init__(self):
        pass

    @abstractmethod
    def calculate_reward(self, graph: GraphGrammar):
        pass


class CalculatorWithConstTorqueOptimization(GraphRewardCalculator):

    def __init__(self,
                 simulation_control,
                 rewarder: SimulationReward,
                 optimization_bounds=(0, 15),
                 optimization_limit=10):
        self.simulation_control = simulation_control
        self.rewarder: SimulationReward = rewarder
        self.bounds = optimization_bounds
        self.limit = optimization_limit

    def simulate_with_control_parameters(self, data, graph):
        return self.simulation_control.run_simulation(graph, data)

    def calculate_reward(self, graph: GraphGrammar):

        def reward_with_parameters(parameters):
            parameters = parameters.round(3)
            data = {"initial_value": parameters}
            sim_output = self.simulate_with_control_parameters(data, graph)
            reward = self.rewarder.calculate_reward(sim_output)
            return -reward

        n_joints = len(get_joint_vector_from_graph(graph))
        if n_joints == 0:
            return (0, [])
        multi_bound = []
        for _ in range(n_joints):
            multi_bound.append(self.bounds)

        result = self.run_optimization(reward_with_parameters, multi_bound)
        return (-result.fun, result.x)

    @abstractmethod
    def run_optimization(self, callback, multi_bound):
        pass


class CalculatorWithOptimizationDirect(CalculatorWithConstTorqueOptimization):

    def run_optimization(self, callback, multi_bound):
        result = direct(callback, multi_bound, maxiter=self.limit)
        return result


class CalculatorWithOptimizationDualAnnealing(CalculatorWithConstTorqueOptimization):

    def run_optimization(self, callback, multi_bound):
        result = dual_annealing(callback, multi_bound, maxiter=self.limit)
        return result


class CalculatorWithGraphOptimization(GraphRewardCalculator):

    def __init__(self, simulation_control, rewarder: SimulationReward, torque_dict):
        self.simulation_control = simulation_control
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
        simulation_output = self.simulation_control.run_simulation(graph, data)
        reward = self.rewarder.calculate_reward(simulation_output)
        return (reward, control_sequence)


class OptimizationParametr(str, Enum):
    START = "START"
    MULTIPLIER = "MULTIPLIER"


class ConstTorqueOptimizationBranchTemplate(GraphRewardCalculator):
    """A template class for constant torque optimization on branches of a graph.
    For use you need implement run_optimization and generate_control_value_on_branch.
    run_optimization for select optimizer.
    generate_control_value_on_branch for generate control.

    Args:
        GraphRewardCalculator: _description_
    """

    def __init__(self,
                 simulation_control,
                 rewarder: SimulationReward,
                 optimization_bounds=(0, 15),
                 optimization_limit=10,
                 select_optimisation_value=OptimizationParametr.START,
                 const_parameter=-0.5):
        self.simulation_control = simulation_control
        self.rewarder: SimulationReward = rewarder
        self.bounds = optimization_bounds
        self.limit = optimization_limit
        self.select_optimisation_value = select_optimisation_value
        self.const_parameter = const_parameter

    def simulate_with_control_parameters(self, data, graph):
        return self.simulation_control.run_simulation(graph, data)

    def extend_parameters_by_const(self, parameters: list[float]) -> list[tuple[float, float]]:
        """"Extends the control parameters list by adding the constant parameter

        Args:
            parameters (list[float]): 

        Raises:
            Exception: _description_

        Returns:
            list[tuple[float, float]]: 
        """
        select_index_order = [0, 0]
        if self.select_optimisation_value == OptimizationParametr.START:
            select_index_order = [0, 1]
        elif self.select_optimisation_value == OptimizationParametr.MULTIPLIER:
            select_index_order = [1, 0]
        else:
            raise Exception("Wrong select_optimisation_value")

        parameters_2d = []
        for param in parameters:
            buf = [0, 0]
            buf[select_index_order[0]] = param
            buf[select_index_order[1]] = self.const_parameter
            parameters_2d.append(tuple(buf))
        return parameters_2d

    def calculate_reward(self, graph: GraphGrammar):

        def reward_with_parameters(parameters):
            parameters_2d = self.extend_parameters_by_const(parameters)
            data = self.generate_control_value_on_branch(graph, parameters_2d)
            sim_output = self.simulate_with_control_parameters(data, graph)
            reward = self.rewarder.calculate_reward(sim_output)
            return -reward

        n_branches = len(joint_root_paths(graph))
        if n_branches == 0:
            return (0, [])
        multi_bound = []
        for _ in range(n_branches):
            multi_bound.append(self.bounds)

        result = self.run_optimization(reward_with_parameters, multi_bound)
        return (-result.fun, result.x)

    @abstractmethod
    def run_optimization(self, callback, multi_bound):
        pass

    @abstractmethod
    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        pass


class LinearControlOptimizationDirect(ConstTorqueOptimizationBranchTemplate):

    def run_optimization(self, callback, multi_bound):
        result = direct(callback, multi_bound, maxiter=self.limit)
        return result

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return linear_control(graph, parameters_2d)


class TendonLikeControlOptimization(ConstTorqueOptimizationBranchTemplate):

    def run_optimization(self, callback, multi_bound):
        result = direct(callback, multi_bound, maxiter=self.limit)
        return result

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return tendon_like_control(graph, parameters_2d)