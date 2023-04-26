import random
import pickle 
import networkx as nx
import matplotlib.pyplot as plt

import pychrono as chrono
from example_vocabulary import (get_terminal_graph_three_finger)

import rostok.virtual_experiment.simulation_step as step

from rostok.criterion.criterion_calc import criterion_calc
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagNotContact, FlagSlipout
from example_vocabulary import (get_terminal_graph_no_joints, get_terminal_graph_three_finger,
                                get_terminal_graph_two_finger, get_terminal_graph_two_finger_exmpl)
from simple_designs import get_two_link_three_finger, get_three_link_two_finger

import rostok.virtual_experiment.simulation_step as step
from rostok.block_builder_api.block_parameters import DefaultFrame, Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
from rostok.trajectory_optimizer.control_optimizer import num_joints
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x
from rostok.trajectory_optimizer.control_optimizer import num_joints
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x

# Constants
MAX_TIME = 2
TIME_STEP = 1e-3

# Graph initialization
# graph = get_terminal_graph_three_finger()
graph = get_three_link_two_finger()
# Create trajectory
nx.draw_networkx(graph,
                    pos=nx.kamada_kawai_layout(graph, dim=2),
                    node_size=800,
                    labels={n: graph.nodes[n]["Node"].label for n in graph})
plt.show()
number_trq = num_joints(graph)
# const_torque_koef = [random.random() for _ in range(number_trq)]
const_torque_koef = [0, 0, 0, 0, -1, 6]
arr_trj = create_torque_traj_from_x(graph, const_torque_koef, MAX_TIME, TIME_STEP)

mat = Material()
mat.Friction = 0.65
mat.DampingF = 0.65

obj = EnvironmentBodyBlueprint(material=mat,
                                        pos=FrameTransform([0, 1, 0],
                                                            [0, -0.048, 0.706, 0.706]))

# Configurate simulation
config_sys = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
flags = [FlagMaxTime(MAX_TIME)]

sim = step.SimulationStepOptimization(arr_trj, graph, obj)
sim.set_flags_stop_simulation(flags)
sim.change_config_system(config_sys)

# Start simulation
sim_output = sim.simulate_system(TIME_STEP, True)

f = open('simout.bin', 'wb')
pickle.dump(sim_output[-1], f)
f.close()

# Weight coefficients for reward function
WEIGHTS = [5, 10, 2]

# Calculate reward value
reward = criterion_calc(sim_output, WEIGHTS)
print(reward)
