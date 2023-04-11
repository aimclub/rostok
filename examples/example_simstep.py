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

mechs = [
    get_terminal_graph_three_finger, get_terminal_graph_no_joints, get_terminal_graph_two_finger
]

for get_graph in mechs:
    # Constants
    MAX_TIME = 2
    TIME_STEP = 1e-3

    graph = get_graph()
    # Create trajectory
    number_trq = num_joints(graph)
    const_torque_koef = [random.random() for _ in range(number_trq)]
    arr_trj = create_torque_traj_from_x(graph, const_torque_koef, MAX_TIME, TIME_STEP)

    # Create object to grasp
    mat = Material()
    mat.Friction = 0.65
    mat.DampingF = 0.65

    obj = EnvironmentBodyBlueprint(material=mat,
                                   pos=FrameTransform([0, 0.3, 0], [0, -0.048, 0.706, 0.706]))
    obj = EnvironmentBodyBlueprint(shape = Box(0.1,0.2,0.5),material=mat,
                                   pos=FrameTransform([0, 0.4, 0], [1,0,0,0]))

    # Configurate simulation
    config_sys = {"Set_G_acc": chrono.ChVectorD(0, -1, 0)}
    flags = [FlagMaxTime(MAX_TIME)]

    sim = step.SimulationStepOptimization(arr_trj, graph, obj,
                                          FrameTransform([0, 0.1, 0], [1, 0, 0, 0]))
    sim.set_flags_stop_simulation(flags)
    sim.change_config_system(config_sys)

    # Start simulation
    sim_output = sim.simulate_system(TIME_STEP, True)

