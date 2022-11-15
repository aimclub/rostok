import context

from engine.node import GraphGrammar
import pychrono as chrono
from engine.robot import Robot
from utils.blocks_utils import NodeFeatures
from utils.simulation_step import SimulationStepOptimization, SimulationDataBlock, SimOut
from scipy.optimize import shgo, dual_annealing, direct
from typing import Callable
from dataclasses import dataclass
from dataclasses import field

@dataclass
class ConfigRewardFunction:
    bound: tuple[float, float] = (-1, 1)
    iters: int = 10
    sim_config: dict[str, str] = field(default_factory=dict)
    time_step: float = 0.001
    time_sim: float = 2
    flags: list = field(default_factory=list)
    criterion_callback: Callable[[SimOut, Robot], float] = None
    get_rgab_object: Callable[[], chrono.ChBody] = None
    params_to_timesiries_array: Callable[[GraphGrammar, list[float]], list] = None


def create_multidimensional_bounds(graph: GraphGrammar, one_d_bound: tuple[float, float]):
    num = num_joints(graph)
    multidimensional_bounds = []
    for i in range(num):
        multidimensional_bounds.append(one_d_bound)

    return multidimensional_bounds


def num_joints(graph: GraphGrammar):
    line_order = graph.get_ids_in_dfs_order()
    list_nodes = list(map(graph.get_node_by_id, line_order))
    return sum(map(NodeFeatures.is_joint, list_nodes))


class ControlOptimizer():
    def __init__(self, cfg: ConfigRewardFunction) -> None:
        self.cfg = cfg

    def create_reward_function(self, generated_graph: GraphGrammar) -> Callable[[list[float]], float]:

        object_to_grab = self.cfg.get_rgab_object()
        init_pos = chrono.ChCoordsysD(object_to_grab.GetCoord())

        def reward(x, is_vis=False):
            # Init object state
            object_to_grab.SetNoSpeedNoAcceleration()
            object_to_grab.SetCoord(init_pos)

            arr_traj = self.cfg.params_to_timesiries_array(generated_graph, x)
            sim = SimulationStepOptimization(
                arr_traj, generated_graph, object_to_grab)
            sim.set_flags_stop_simulation(self.cfg.flags)
            sim.change_config_system(self.cfg.sim_config)
            sim_output = sim.simulate_system(self.cfg.time_step, is_vis)
            rew = self.cfg.criterion_callback(sim_output, sim.grab_robot)

            object_to_grab.SetNoSpeedNoAcceleration()
            object_to_grab.SetCoord(init_pos)

            return rew

        return reward

    def start_optimisation(self, generated_graph: GraphGrammar) -> tuple[float, float]:
        reward_fun = self.create_reward_function(generated_graph)
        multi_bound = create_multidimensional_bounds(
            generated_graph, self.cfg.bound)
        result = direct(reward_fun, multi_bound, maxiter=self.cfg.iters)
        return (result.fun, result.x)
