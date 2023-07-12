from abc import abstractmethod

from scipy.optimize import direct, dual_annealing, shgo

from rostok.criterion.criterion_calculation import SimulationReward
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph


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
                 optimization_bounds=(6, 15),
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
