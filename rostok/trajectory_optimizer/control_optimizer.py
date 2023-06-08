from scipy.optimize import direct, dual_annealing, shgo

from rostok.criterion.criterion_calculation import SimulationReward
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import (NodeFeatures, get_joint_vector_from_graph)

# @dataclass
# class ConfigRewardFunction:
#     """
#     Attributes:
#         bound: tuple (lower bound, upper bound) extend to joints number
#         iters: number of iteration optimization algorithm
#         sim_config: config passed to Chrono engine
#         time_step: simulation step
#         time_sim: simulation duration
#         flags: List of stop flags, breaks sim
#         criterion_callback: calls after simulation (SimOut, Robot) -> float
#         get_rgab_object: calls before simulation () -> ObjectToGrasp
#         params_to_timesiries_array: calls before simulation to calculate trajectory
#             (GraphGrammar, list[float]) -> list[list] in dfs form, See class
#             SimulationStepOptimization
#     """
#     bound: tuple[float, float] = (-1, 1)
#     iters: int = 20
#     sim_config: dict[str, str] = field(default_factory=dict)
#     time_step: float = 0.005
#     time_sim: float = 2
#     flags: list = field(default_factory=list)
#     criterion_callback: Callable[[SimOut, Robot], float] = None
#     get_rgab_object_callback: Callable[[], chrono.ChBody] = None
#     params_to_timesiries_callback: Callable[[GraphGrammar, list[float]], list] = None


class GraphRewardCounter:

    def __init__(self):
        pass

    def count_reward(self, graph: GraphGrammar):
        pass


class CounterWithOptimization(GraphRewardCounter):

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

    def count_reward(self, graph: GraphGrammar):

        def reward_with_parameters(parameters):
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

        result = dual_annealing(reward_with_parameters, multi_bound, maxiter=self.limit)
        return (-result.fun, result.x)


class CounterWithOptimizationDirect(GraphRewardCounter):

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

    def count_reward(self, graph: GraphGrammar):

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

        result = direct(reward_with_parameters, multi_bound, maxiter=self.limit)
        return (-result.fun, result.x.round(3))


class CounterGraphOptimization(GraphRewardCounter):

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

    def count_reward(self, graph: GraphGrammar):

        n_joints = get_joint_vector_from_graph(graph)
        if n_joints == 0:
            return (0, [])
        control_sequence = self.build_control_from_graph(graph)
        data = {"initial_value": control_sequence}
        simulation_output = self.simulation_control.run_simulation(graph, data, True)
        reward = self.rewarder.calculate_reward(simulation_output)
        return (reward, control_sequence)
