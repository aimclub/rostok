import random

import pychrono as chrono
from example_vocabulary import (B_NODES, J_NODES, LM_MOUNTS, RM_MOUNTS,
                                get_terminal_graph_three_finger)

import rostok.virtual_experiment.simulation_step as step
from rostok.block_builder.basic_node_block import SimpleBody
from rostok.block_builder.node_render import (ChronoBodyEnv, DefaultChronoMaterial, FrameTransform)
from rostok.criterion.criterion_calc import criterion_calc
from rostok.criterion.flags_simualtions import FlagMaxTime
from rostok.graph_grammar.node import BlockWrapper
from rostok.graph_grammar.nodes_division import nodes_division, sort_left_right
from rostok.trajectory_optimizer.control_optimizer import num_joints
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x

# Constants
MAX_TIME = 1
TIME_STEP = 1e-3

# Graph initialization
graph = get_terminal_graph_three_finger()

# Create trajectory
number_trq = num_joints(graph)
const_torque_koef = [random.random() for _ in range(number_trq)]
arr_trj = create_torque_traj_from_x(graph, const_torque_koef, MAX_TIME, TIME_STEP)

# Create object to grasp with material props
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

# Weight coefficients for reward function
WEIGHTS = [5, 1, 1, 5]
# Time to grasp object
GAIT_PERIOD = 2.5

#Applying supportive functions to division nodes of graph onto different types
J_NODES_NEW = nodes_division(sim.grab_robot, J_NODES)
B_NODES_NEW = nodes_division(sim.grab_robot, B_NODES)
RB_NODES_NEW = sort_left_right(sim.grab_robot, RM_MOUNTS, B_NODES)
LB_NODES_NEW = sort_left_right(sim.grab_robot, LM_MOUNTS, B_NODES)
RJ_NODES_NEW = sort_left_right(sim.grab_robot, RM_MOUNTS, J_NODES)
LJ_NODES_NEW = sort_left_right(sim.grab_robot, LM_MOUNTS, J_NODES)

# Calculate reward value
reward = criterion_calc(sim_output, B_NODES_NEW, J_NODES_NEW, LB_NODES_NEW, RB_NODES_NEW, WEIGHTS,
                        GAIT_PERIOD)
print(reward)