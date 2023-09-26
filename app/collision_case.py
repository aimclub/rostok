import random
import hyperparameters as hp

from mcts_run_setup import config_with_standard_graph
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from rostok.library.obj_grasp.objects import get_object_box, get_object_ellipsoid, get_object_parametrized_sphere
from rostok.library.rule_sets import ruleset_old_style_graph
from rostok.library.rule_sets.simple_designs import (get_two_link_three_finger)
from rostok.simulation_chrono.basic_simulation import SimulationResult

# create blueprint for object to grasp

rules, torque_dict = ruleset_old_style_graph.create_rules()

grasp_object_blueprint = get_object_parametrized_sphere(0.5)
grasp_object_blueprint = get_object_box(1.2, 0.5, 0.8, 0)

# create reward counter using run setup function
control_optimizer = config_with_standard_graph(grasp_object_blueprint, torque_dict)

print("Object to grasp:", grasp_object_blueprint.shape)
dict = {
    item: getattr(hp, item)
    for item in dir(hp)
    if not item.startswith("__") and not item.endswith("__")
}
for key, value in dict.items():
    print(key, value)
simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_scenario

graph = get_two_link_three_finger()

controll_parameters = control_optimizer.build_control_from_graph(graph)
 

controll_parameters = {"initial_value": controll_parameters}
controll_parameters = {"initial_value": [300, 100, 300, 100, 300, 100]}
vis = True
#simulation_output: SimulationResult = simulation_manager.run_simulation(graph, data, [[-45.0, 0.0],[-45,0],[-45,0]], vis, True)
simulation_output: SimulationResult = simulation_manager.run_simulation(
    graph, controll_parameters, vis, True)

res = simulation_rewarder.calculate_reward(simulation_output)
print('reward', res)