from dataclasses import dataclass, field
from typing import Callable

import pychrono as chrono
from scipy.optimize import direct, shgo, dual_annealing
from functools import partial
from rostok.block_builder.blocks_utils import NodeFeatures
from rostok.graph_grammar.node import GraphGrammar
from rostok.virtual_experiment.robot import Robot
from rostok.virtual_experiment.simulation_step import (SimOut, SimulationStepOptimization)
from typing import Union
import time

@dataclass
class _ConfigRewardFunction:
    """
    Attributes:
        sim_config: config passed to Chrono engine
        time_step: simulation step
        time_sim: simulation duration
        flags: List of stop flags, breaks sim
        criterion_callback: calls after simulation (SimOut, Robot) -> float
        get_rgab_object: calls before simulation () -> ObjectToGrasp
        params_to_timesiries_array: calls before simulation to calculate trajectory
            (GraphGrammar, list[float]) -> list[list] in dfs form, See class SimulationStepOptimization
    """
    sim_config: dict[str, str] = field(default_factory=dict)
    time_step: float = 0.001
    time_sim: float = 2
    flags: list = field(default_factory=list)
    criterion_callback: Callable[[SimOut, Robot], float] = None
    get_rgab_object_callback: Callable[[], chrono.ChBody] = None
    params_to_timesiries_callback: Callable[[GraphGrammar, list[float]], list] = None


class ConfigVectorJoints(_ConfigRewardFunction):
    """
    Length of vector X equal number of joints
    Attributes:
        bound: tuple (lower bound, upper bound) extend to joints number
        iters: number of iteration optimization algorithm
    """
    bound: tuple[float, float] = (-1, 1)
    iters: int = 10
    optimizer_scipy = partial(direct)


class ConfigGraphControl(_ConfigRewardFunction):
    pass


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

    def __init__(self, cfg: Union[ConfigGraphControl, ConfigVectorJoints]) -> None:
        self.cfg = cfg

    def create_reward_function(self,
                               generated_graph: GraphGrammar,
                               is_vis=False,
                               is_debug=False) -> Callable[[float], float]:
        """Create reward function
        Args:
            generated_graph (GraphGrammar):
        Returns:
            Callable[[list[float]], float]: Function of virtual experemnt that
            returns reward based on criterion_callback
        """

        def reward(x, is_vis=is_vis, is_debug=is_debug):
            start_time = time.time()
            
            # Init object state
            object_to_grab = self.cfg.get_rgab_object_callback()
            arr_traj = self.cfg.params_to_timesiries_callback(generated_graph, x)
            sim = SimulationStepOptimization(arr_traj, generated_graph, object_to_grab)
            sim.set_flags_stop_simulation(self.cfg.flags)
            sim.change_config_system(self.cfg.sim_config)
            sim_output = sim.simulate_system(self.cfg.time_step, is_vis)
            rew = self.cfg.criterion_callback(sim_output)
            
            elapsed = time.time() - start_time
            
            if is_debug:
                print('Rew:', rew, 'Vec:', x, "Elapsed t:", elapsed)
            return rew

        return reward

    def start_optimisation(self,
                           generated_graph: GraphGrammar,
                           is_debug=False) -> tuple[float, list[float]]:
        """Start find optimal control. If graph-based control is used, 
        then run one simulation. If torque control search is used, 
        it starts the optimization process.
        Parameters
        ----------
        generated_graph : GraphGrammar
        Returns
        -------
        tuple[float, list[float]]
        Reward, values for generate control
        """
        if isinstance(self.cfg, ConfigVectorJoints):
            reward_fun = self.create_reward_function(generated_graph, is_debug=is_debug)
            multi_bound = create_multidimensional_bounds(generated_graph, self.cfg.bound)
            if len(multi_bound) == 0:
                return (0, 0)
            result = self.cfg.optimizer_scipy(reward_fun, multi_bound, maxiter=self.cfg.iters)
            return (result.fun, result.x)
        elif isinstance(self.cfg, ConfigGraphControl):
            n_joint = num_joints(generated_graph)
            if n_joint == 0:
                return (0, 0)
            reward_fun = self.create_reward_function(generated_graph, is_debug=is_debug)
            unused_list = [0 for t in range(n_joint)]
            res = reward_fun(unused_list)
            return (res, unused_list)