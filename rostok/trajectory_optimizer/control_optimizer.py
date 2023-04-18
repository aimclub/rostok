from dataclasses import dataclass, field
from typing import Callable

import pychrono as chrono
from scipy.optimize import direct, dual_annealing, shgo

from rostok.graph_grammar.node_block_typing import NodeFeatures
from rostok.graph_grammar.node import GraphGrammar
from rostok.virtual_experiment.robot import Robot
from rostok.virtual_experiment.simulation_step import (SimOut, SimulationStepOptimization)


@dataclass
class ConfigRewardFunction:
    """
    Attributes:
        bound: tuple (lower bound, upper bound) extend to joints number
        iters: number of iteration optimization algorithm
        sim_config: config passed to Chrono engine
        time_step: simulation step
        time_sim: simulation duration
        flags: List of stop flags, breaks sim
        criterion_callback: calls after simulation (SimOut, Robot) -> float
        get_rgab_object: calls before simulation () -> ObjectToGrasp
        params_to_timesiries_array: calls before simulation to calculate trajectory
            (GraphGrammar, list[float]) -> list[list] in dfs form, See class SimulationStepOptimization
    """
    bound: tuple[float, float] = (-1, 1)
    iters: int = 20
    sim_config: dict[str, str] = field(default_factory=dict)
    time_step: float = 0.005
    time_sim: float = 2
    flags: list = field(default_factory=list)
    criterion_callback: Callable[[SimOut, Robot], float] = None
    get_rgab_object_callback: Callable[[], chrono.ChBody] = None
    params_to_timesiries_callback: Callable[[GraphGrammar, list[float]], list] = None


def create_multidimensional_bounds(graph: GraphGrammar, one_d_bound: tuple[float, float]):
    num = num_joints(graph)
    multidimensional_bounds = []
    for _ in range(num):
        multidimensional_bounds.append(one_d_bound)

    return multidimensional_bounds


def num_joints(graph: GraphGrammar) -> int:
    """ Detect joints based on :py:mod:`node_render` clases

    Args:
        graph (GraphGrammar):

    Returns:
        int: number of joints
    """
    line_order = graph.get_ids_in_dfs_order()
    list_nodes = list(map(graph.get_node_by_id, line_order))
    return sum(map(NodeFeatures.is_joint, list_nodes))


class ControlOptimizer():

    def __init__(self, cfg: ConfigRewardFunction) -> None:
        self.cfg = cfg

    def create_reward_function(self,
                               generated_graph: GraphGrammar) -> Callable[[list[float]], float]:
        """Create reward function

        Args:
            generated_graph (GraphGrammar):

        Returns:
            Callable[[list[float]], float]: Function of virtual experemnt that
            returns reward based on criterion_callback
        """

        def reward(x, is_vis=False):
            # Init object state
            x = [round(elem, 3) for elem in x]
            object_to_grab = self.cfg.get_rgab_object_callback()
            arr_traj = self.cfg.params_to_timesiries_callback(generated_graph, x)
            sim = SimulationStepOptimization(arr_traj, generated_graph, object_to_grab)
            sim.set_flags_stop_simulation(self.cfg.flags)
            sim.change_config_system(self.cfg.sim_config)
            sim_output = sim.simulate_system(self.cfg.time_step, is_vis)
            rew = self.cfg.criterion_callback(sim_output)
            return rew

        return reward

    def start_optimisation(self, generated_graph: GraphGrammar) -> tuple[float, list[float]]:

        reward_fun = self.create_reward_function(generated_graph)
        multi_bound = create_multidimensional_bounds(generated_graph, self.cfg.bound)
        if len(multi_bound) == 0:
            return (0, 0)
        result = dual_annealing(reward_fun, multi_bound, maxiter=self.cfg.iters)
        return (result.fun, result.x)
