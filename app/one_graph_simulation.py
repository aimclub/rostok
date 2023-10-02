from mcts_run_setup import config_independent_torque, config_with_tendon

from rostok.library.obj_grasp.objects import get_object_sphere
from rostok.library.rule_sets.simple_designs import (
    get_three_link_one_finger, get_three_link_one_finger_independent)
from rostok.simulation_chrono.simulation_utils import SimulationResult

# create blueprint for object to grasp
grasp_object_blueprint = get_object_sphere(0.05)

# create reward counter using run setup function
# control_optimizer = config_with_const_troques(grasp_object_blueprint)
control_optimizer = config_independent_torque(grasp_object_blueprint)
control_optimizer = config_with_tendon(grasp_object_blueprint)

simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_scenario

graph = get_three_link_one_finger_independent()
graph = get_three_link_one_finger()
control = [10]

data = control_optimizer.optim_parameters2data_control(control, graph)

vis = True

#simulation_output: SimulationResult = simulation_manager.run_simulation(graph, data, [[-45.0, 0.0],[-45,0],[-45,0]], vis, True)
simulation_output: SimulationResult = simulation_manager.run_simulation(graph, data, [[0.0, 0, 0]], vis, True)

res = simulation_rewarder.calculate_reward(simulation_output)
print('reward', res)
