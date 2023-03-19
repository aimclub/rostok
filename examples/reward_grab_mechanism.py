import random
import pickle 

import pychrono as chrono
from example_vocabulary import (get_terminal_graph_three_finger)

import rostok.virtual_experiment.simulation_step as step
from rostok.block_builder.envbody_shapes import Box
from rostok.block_builder.node_render import (ChronoBodyEnv, DefaultChronoMaterial, FrameTransform)
from rostok.criterion.criterion_calc import criterion_calc
from rostok.virtual_experiment.flags_simualtions import FlagMaxTime, FlagNotContact, FlagSlipout
from rostok.graph_grammar.node import BlockWrapper
from rostok.robot.nodes_division import nodes_division, sort_left_right
from rostok.trajectory_optimizer.control_optimizer import num_joints
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x

# Constants
MAX_TIME = 2
TIME_STEP = 1e-3

# Graph initialization
graph = get_terminal_graph_three_finger()

# Create trajectory
number_trq = num_joints(graph)
# const_torque_koef = [random.random() for _ in range(number_trq)]
const_torque_koef = [0, 0, 0, 0, -1, 6]
arr_trj = create_torque_traj_from_x(graph, const_torque_koef, MAX_TIME, TIME_STEP)


# Create object to grasp with material props
mat = DefaultChronoMaterial()
mat.Friction = 0.65
mat.DampingF = 0.65
obj = BlockWrapper(ChronoBodyEnv,
                   shape=Box(),
                   material=mat,
                   pos=FrameTransform([0, 0.4, 0], [0, -0.048, 0.706, 0.706]))

# Configurate simulation
config_sys = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
flags = [FlagMaxTime(MAX_TIME), FlagNotContact(1), FlagSlipout(0.5, 0.5)]

sim = step.SimulationStepOptimization(arr_trj, graph, obj)
sim.set_flags_stop_simulation(flags)
sim.change_config_system(config_sys)

# Start simulation
sim_output = sim.simulate_system(TIME_STEP, True)

#Save dictionary
f = open('simout.bin', 'wb')
pickle.dump(sim_output[-1], f)
f.close()

# Weight coefficients for reward function
WEIGHTS = [5, 10, 2]


# Calculate reward value
reward = criterion_calc(sim_output, WEIGHTS)
print(reward)

