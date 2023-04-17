import random

import pychrono as chrono
from example_vocabulary import (get_terminal_graph_no_joints, get_terminal_graph_three_finger,
                                get_terminal_graph_two_finger)

import rostok.virtual_experiment.simulation_step as step
from rostok.block_builder_api.block_parameters import DefaultFrame, Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
from rostok.trajectory_optimizer.control_optimizer import num_joints
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.simulation_chrono.basic_simulation import SystemPreview
mechs = [
    get_terminal_graph_three_finger, get_terminal_graph_no_joints, get_terminal_graph_two_finger
]

for get_graph in mechs:
    # Constants
    MAX_TIME = 2
    TIME_STEP = 1e-3

    graph = get_graph()
    sim = SystemPreview()

    # Create object to grasp
    mat = Material()
    mat.Friction = 0.65
    mat.DampingF = 0.65
    obj = EnvironmentBodyBlueprint(shape = Box(0.1,0.2,0.5),material=mat,
                                   pos=FrameTransform([0, -0.4, 0], [1,0,0,0]))
    sim.add_object(obj)
    sim.add_design(graph)
    sim.simulate(10000, True)
