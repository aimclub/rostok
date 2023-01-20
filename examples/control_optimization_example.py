import example_vocabulary
import pychrono as chrono

import rostok.criterion.criterion_calc as criterion
from rostok.block_builder.envbody_shapes import Cylinder
from rostok.block_builder.node_render import (ChronoBodyEnv,
                                              DefaultChronoMaterial,
                                              FrameTransform)
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagNotContact
from rostok.graph_grammar.node import BlockWrapper, GraphGrammar, Node
from rostok.trajectory_optimizer.control_optimizer import (
    ConfigRewardFunction, ControlOptimizer)
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x
from rostok.virtual_experiment.simulation_step import SimOut

# Init grasping object
def get_object_to_grasp():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=Cylinder(),
                       material=matich,
                       pos=FrameTransform([0, 1, 0], [0, -0.048, 0.706, 0.706]))

    return obj

# Calculate criterion of grabing
def grab_crtitrion(sim_output: dict[int, SimOut], grab_robot, node_feature: list[list[Node]], gait,
                   weight):
    j_nodes = criterion.nodes_division(grab_robot, node_feature[1])
    b_nodes = criterion.nodes_division(grab_robot, node_feature[0])
    rb_nodes = criterion.sort_left_right(grab_robot, node_feature[3], node_feature[0])
    lb_nodes = criterion.sort_left_right(grab_robot, node_feature[2], node_feature[0])

    return criterion.criterion_calc(sim_output, b_nodes, j_nodes, rb_nodes, lb_nodes, weight, gait)

# Create criterion function
def create_grab_criterion_fun(node_features, gait, weight):

    def fun(sim_output, grab_robot):
        return grab_crtitrion(sim_output, grab_robot, node_features, gait, weight)

    return fun

# Create torque trajectory function
def create_traj_fun(stop_time: float, time_step: float):

    def fun(graph: GraphGrammar, x: list[float]):
        return create_torque_traj_from_x(graph, x, stop_time, time_step)

    return fun



GAIT = 2.5
WEIGHT = [5, 0, 1, 5]

# Init configuration of control optimizing
cfg = ConfigRewardFunction()
cfg.bound = (-5, 5)
cfg.iters = 2
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.005
cfg.time_sim = 2
cfg.flags = [FlagMaxTime(cfg.time_sim)]
"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(example_vocabulary.NODE_FEATURES, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

# Init control optimization
control_optimizer = ControlOptimizer(cfg)
graph = example_vocabulary.get_terminal_graph_three_finger()

# Run optimization
res = control_optimizer.start_optimisation(graph)
print(res)