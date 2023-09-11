import hyperparameters as hp
import matplotlib.pyplot as plt
import pickle

import numpy as np
from mcts_run_setup import config_combination_force_tendon_multiobject

import rostok.library.obj_grasp.objects as obj_grasp
import rostok.library.rule_sets.simple_designs as des
from rostok.simulation_chrono.basic_simulation import SimulationResult
from rostok.graph_grammar.graph_utils import plot_graph
object_blueprint = []
# create blueprint for object to grasp
# object_blueprint.append(obj_grasp.get_object_parametrized_trapezohedron(0.15, mass=0.467))
# object_blueprint.append(obj_grasp.get_object_cylinder(0.155/2, 0.155, 0, mass = 0.261))
# object_blueprint.append(obj_grasp.get_object_box(0.14, 0.19, 0.28, 0, mass = 0.268))
object_blueprint.append(obj_grasp.get_object_parametrized_dipyramid_3(0.1, 0.13, 90))
object_blueprint.append(obj_grasp.get_object_ellipsoid(0.14, 0.14, 0.22, 0, mass = 0.188))

# create reward counter using run setup function
control_optimizer = config_combination_force_tendon_multiobject(object_blueprint, [1, 1, 1, 1, 1])

dict = {
    item: getattr(hp, item)
    for item in dir(hp)
    if not item.startswith("__") and not item.endswith("__")
}
for key, value in dict.items():
    print(key, value)
simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_scenario

graph = des.get_two_link_three_finger()
# graph = des.get_two_same_link_one_finger()
graph = des.get_four_same_link_one_finger()
graph = des.get_three_same_link_one_finger()
graph = des.get_three_same_link_one_finger()
graph = des.get_four_same_link_one_finger()
graph = des.get_five_same_link_one_finger()
graph = des.get_six_same_link_one_finger()
graph = des.get_seven_same_link_one_finger()


control = [[10,10,10,10]]

data = control_optimizer.optim_parameters2data_control(control, graph)[0]

vis = True
#simulation_output: SimulationResult = simulation_manager.run_simulation(graph, data, [[-45.0, 0.0],[-45,0],[-45,0]], vis, True)
full_reward = 0
for sim_scen in simulation_manager:
    simulation_output: SimulationResult = sim_scen[0].run_simulation(
        graph, data, [[-45.0, 0, 0, 0, 0, 0, 0], [-45.0, 0, 0, 0, 0, 0, 0], [-45.0, 0, 0, 0, 0, 0, 0], [-45.0, 0, 0, 0, 0, 0, 0]], vis, True)
    res = simulation_rewarder.calculate_reward(simulation_output)
    full_reward += res
    print('reward', res)

print("full_reward", full_reward)

