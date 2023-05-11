import numpy as np

from rostok.simulation_chrono.basic_simulation import ConstTorqueGrasp
from rostok.criterion.simulation_flags import FlagContactTimeOut, FlagFlyingApart, FlagSlipout
from simple_designs import get_three_link_one_finger_with_no_control, get_two_link_one_finger
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from control_optimisation import get_object_to_grasp

# construct a simulation manager
simulation_control = ConstTorqueGrasp(0.005, 3)
# add object to grasp
simulation_control.grasp_object_callback = get_object_to_grasp
# create flags
simulation_control.add_flag(FlagContactTimeOut(2))
simulation_control.add_flag(FlagFlyingApart(10))
simulation_control.add_flag(FlagSlipout(1.5))

graph = get_two_link_one_finger()
print(get_joint_vector_from_graph(graph))
control = np.random.random(len(get_joint_vector_from_graph(graph)))
print(control)
data = {"initial_value": list(control)}
simulation_control.run_simulation(graph, data)


