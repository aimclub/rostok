from abc import abstractmethod, ABC
import numpy as np

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
    
    @abstractmethod
    def print_log(self):
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


class OptimizationParametr(Enum):
    """Enum for select optimization value.
    Start means constant part of linear function, multiplier 
    means slope of line.
    """
    START = 1
    MULTIPLIER = 2


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


class ConstControlOptimizationDirect(ConstTorqueOptimizationBranchTemplate, ABC):
    """A template class for constant torque optimization on branches of a graph
    with direct optimization.  

    Args:
        ConstTorqueOptimizationBranchTemplate (_type_): _description_
    """

    def run_optimization(self, callback, multi_bound):
        result = direct(callback, multi_bound, maxiter=self.limit)
        return result


class LinearControlOptimizationDirect(ConstControlOptimizationDirect):

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return linear_control(graph, parameters_2d)


class TendonLikeControlOptimization(ConstControlOptimizationDirect):

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return tendon_like_control(graph, parameters_2d)
    

# ==================================
# Prototype Class MultiObject Search
# ==================================


# Prototype 1 - Idea 1 - list[BlueprintObject] - One Control to Grasp Them All
class ConstTorqueMultiOptimizationBranchTemplate(GraphRewardCalculator):
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
                 const_parameter=-0.5,
                 object_weights = [1,1,1]):
        self.simulation_control = simulation_control
        self.rewarder: SimulationReward = rewarder
        self.bounds = optimization_bounds
        self.limit = optimization_limit
        self.select_optimisation_value = select_optimisation_value
        self.const_parameter = const_parameter
        self.object_weights = object_weights

    def simulate_with_control_parameters(self, data, graph, id_object = 0):
        return self.simulation_control.run_simulation(graph, data, id_object=id_object)

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

        def reward_with_parameters(parameters, id_object):
            parameters_2d = self.extend_parameters_by_const(parameters)
            data = self.generate_control_value_on_branch(graph, parameters_2d)
            sim_output = self.simulate_with_control_parameters(data, graph, id_object)
            

            reward = self.rewarder.calculate_reward(sim_output)
            print(id_object)
            return -reward

        n_branches = len(joint_root_paths(graph))
        if n_branches == 0:
            return (0, [])
        multi_bound = []
        for _ in range(n_branches):
            multi_bound.append(self.bounds)

        results = []
        for id in range(len(self.object_weights)):
            results.append(self.run_optimization(reward_with_parameters, multi_bound, (id,)))
        return results #(-result.fun, result.x)

    @abstractmethod
    def run_optimization(self, callback, multi_bound, args):
        pass

    @abstractmethod
    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        pass


class LinearControlMultiOptimizationDirect(ConstTorqueMultiOptimizationBranchTemplate):

    def run_optimization(self, callback, multi_bound,args):
        result = direct(callback, multi_bound, maxiter=self.limit, args=args)
        return result

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return linear_control(graph, parameters_2d)


class TendonLikeControlMultiOptimization(ConstTorqueMultiOptimizationBranchTemplate):

    def run_optimization(self, callback, multi_bound,args):
        result = direct(callback, multi_bound, maxiter=self.limit, args=args)
        return result

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return tendon_like_control(graph, parameters_2d)
    

# ==================================
# Prototype Class MultiObject Search
# ==================================


# Prototype 1 - Idea 1 - list[BlueprintObject] - Many Control to Many Objects
class ConstTorqueMultiOptimizationBranchTemplate(GraphRewardCalculator):
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
                 const_parameter=-0.5,
                 object_weights = None):
        self.simulation_control = simulation_control
        self.rewarder: SimulationReward = rewarder
        self.bounds = optimization_bounds
        self.limit = optimization_limit
        self.select_optimisation_value = select_optimisation_value
        self.const_parameter = const_parameter
        self.object_weights = object_weights
        
        if not self.object_weights:
            id = 0
            try:
                while True:
                    self.simulation_control.grasp_object_callback(id)
                    id +=1
            except IndexError:
                self.object_weights = [1/(id + 1) for __ in range(id)]

    def simulate_with_control_parameters(self, data, graph, id_object = 0):
        return self.simulation_control.run_simulation(graph, data, id_object=id_object)

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

        def reward_with_parameters(parameters, id_object):
            parameters_2d = self.extend_parameters_by_const(parameters)
            data = self.generate_control_value_on_branch(graph, parameters_2d)
            sim_output = self.simulate_with_control_parameters(data, graph, id_object)
            

            reward = self.rewarder.calculate_reward(sim_output)
            return -reward

        n_branches = len(joint_root_paths(graph))
        if n_branches == 0:
            return (0, [])
        multi_bound = []
        for _ in range(n_branches):
            multi_bound.append(self.bounds)

        results = []
        for id in range(len(self.object_weights)):
            results.append(self.run_optimization(reward_with_parameters, multi_bound, (id,)))

        l_object_callback = self.simulation_control.grasp_object_callback
        final_reward = 0
        controllers = {}
        for id, res, w in zip(range(len(results)), results, self.object_weights):
            final_reward += res.fun*w
            controllers[l_object_callback(id)] = res.x
        return (-final_reward, controllers)

    @abstractmethod
    def run_optimization(self, callback, multi_bound, args):
        pass

    @abstractmethod
    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        pass


class LinearControlMultiOptimizationDirect(ConstTorqueMultiOptimizationBranchTemplate):

    def run_optimization(self, callback, multi_bound,args):
        result = direct(callback, multi_bound, maxiter=self.limit, args=args)
        return result

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return linear_control(graph, parameters_2d)


class TendonLikeControlMultiOptimization(ConstTorqueMultiOptimizationBranchTemplate):

    def run_optimization(self, callback, multi_bound,args):
        result = direct(callback, multi_bound, maxiter=self.limit, args=args)
        return result

    def generate_control_value_on_branch(self, graph: GraphGrammar,
                                         parameters_2d: list[tuple[float, float]]):
        return tendon_like_control(graph, parameters_2d)