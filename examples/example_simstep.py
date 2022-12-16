import random

import numpy as np
import pychrono as chrono
from example_vocabulary import (get_terminal_graph_no_joints,
                                get_terminal_graph_three_finger,
                                get_terminal_graph_two_finger)

import rostok.virtual_experiment.simulation_step as step
from rostok.block_builder.basic_node_block import SimpleBody
from rostok.block_builder.node_render import (ChronoBodyEnv,
                                              DefaultChronoMaterial,
                                              FrameTransform)
from rostok.criterion.flags_simualtions import FlagMaxTime
from rostok.graph_grammar.node import BlockWrapper
from rostok.trajectory_optimizer.control_optimizer import num_joints
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x


mechs = [
    get_terminal_graph_three_finger, get_terminal_graph_no_joints, get_terminal_graph_two_finger
]

for get_graph in mechs:
    # Constants
    MAX_TIME = 1
    TIME_STEP = 1e-3

    graph = get_graph()
    
    # Create trajectory 
    number_trq = num_joints(graph)
    const_torque_koef = [random.random() for _ in range(number_trq)]
    arr_trj = create_torque_traj_from_x(graph, const_torque_koef, MAX_TIME, TIME_STEP)

    # Create object to grasp
    mat = DefaultChronoMaterial()
    mat.Friction = 0.65
    mat.DampingF = 0.65
    obj = BlockWrapper(ChronoBodyEnv,
                        shape=SimpleBody.BOX,
                        material=mat,
                        pos=FrameTransform([0, 1, 0], [0, -0.048, 0.706, 0.706]))

    # Configurate simulation
    config_sys = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
    flags = [FlagMaxTime(MAX_TIME)]
    
    sim = step.SimulationStepOptimization(arr_trj, graph, obj)
    sim.set_flags_stop_simulation(flags)
    sim.change_config_system(config_sys)
    
    # Start simulation
    sim_output = sim.simulate_system(TIME_STEP, True)
