import example_vocabulary
import pychrono as chrono

import rostok.criterion.criterion_calc as criterion
from rostok.block_builder_chrono.block_classes import (ChronoEasyShapeObject, DefaultChronoMaterial,
                                                       FrameTransform)
from rostok.block_builder_chrono.easy_body_shapes import Cylinder
from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact, FlagSlipout)
from rostok.graph.node import BlockWrapper, Node
from rostok.graph_grammar.graph_grammar import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (ConfigRewardFunction, ControlOptimizer)
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x
from rostok.virtual_experiment.simulation_step import SimOut


# Init grasping object
def get_object_to_grasp():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=Cylinder(),
                       material=matich,
                       pos=FrameTransform([0, 0.6, 0], [0, -0.048, 0.706, 0.706]))

    return obj


# Calculate criterion of grabing
def grab_crtitrion(sim_output: dict[int, SimOut], weight):

    return criterion.criterion_calc(sim_output, weight)


# Create criterion function
def create_grab_criterion_fun(weight):

    def fun(sim_output):
        return grab_crtitrion(sim_output, weight)

    return fun


# Create torque trajectory function
def create_traj_fun(stop_time: float, time_step: float):

    def fun(graph: GraphGrammar, x: list[float]):
        return create_torque_traj_from_x(graph, x, stop_time, time_step)

    return fun


WEIGHT = [5, 10, 2]

# Init configuration of control optimizing
cfg = ConfigRewardFunction()
cfg.bound = (-7, 7)
cfg.iters = 20
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.005
cfg.time_sim = 3
cfg.flags = [FlagMaxTime(cfg.time_sim), FlagNotContact(1), FlagSlipout(0.5, 0.5)]
"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(WEIGHT)
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

# Print result with visualisation
rew_func = control_optimizer.create_reward_function(graph)
rew_func(res[1], True)
pass