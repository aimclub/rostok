
import hyperparameters as hp
import matplotlib.pyplot as plt

import numpy as np
from mcts_run_setup import config_with_const_troques, config_with_tendon

from rostok.library.obj_grasp.objects import (get_object_ellipsoid, get_object_box, get_object_parametrized_dipyramid_3, get_object_sphere)

from rostok.library.rule_sets.simple_designs import (
    get_two_link_three_finger, get_two_same_link_one_finger, get_four_same_link_one_finger,
    get_three_same_link_one_finger, get_two_link_three_finger_rotated, get_three_link_three_finger)
from rostok.simulation_chrono.basic_simulation import SimulationResult

# create blueprint for object to grasp
grasp_object_blueprint = get_object_sphere(0.05)
# grasp_object_blueprint = get_object_ellipsoid(0.14, 0.14, 0.22, 0, mass = 0.188)
# grasp_object_blueprint = get_object_box(0.155, 0.127, 0.088*2, 0, mass = 0.176)
# grasp_object_blueprint = get_object_box(0.146, 0.147,0.25, 0, mass=0.164)
# grasp_object_blueprint = get_object_parametrized_dipyramid_3(0.1, 0.167, 90)


# create reward counter using run setup function
# control_optimizer = config_with_const_troques(grasp_object_blueprint)
control_optimizer = config_with_tendon(grasp_object_blueprint)

# print("Object to grasp:", grasp_object_blueprint.shape)
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
# graph = get_two_same_link_one_finger()
# graph = get_four_same_link_one_finger()
graph = get_three_same_link_one_finger()
# graph = get_two_link_three_finger_rotated()
# graph = get_three_link_three_finger()
control = [15, 15, 15]

data = control_optimizer.optim_parameters2data_control(control, graph)
control_optimizer.data.create_pulley_data_file = True
vis = True

#simulation_output: SimulationResult = simulation_manager.run_simulation(graph, data, [[-45.0, 0.0],[-45,0],[-45,0]], vis, True)
simulation_output: SimulationResult = simulation_manager.run_simulation(
    graph, data, [[-25.0, 0, 0, 0], [-45, 0, 0], [-45, 0, 0]], vis, True)
if not vis:
    fig = plt.figure(figsize=(12, 5))
    time_vector = simulation_output.time_vector
    velocity_data_idx = list(simulation_output.robot_final_ds.get_data("body_velocity").keys())
    trajectories = simulation_output.robot_final_ds.get_data("COG")[velocity_data_idx[-1]]
    velocity_data = simulation_output.robot_final_ds.get_data("body_velocity")[velocity_data_idx[-2]]
    velocity_data = [np.linalg.norm(x) for x in velocity_data]
    #velocity_data = [x[0] for x in velocity_data]
    force_data = simulation_output.environment_final_ds.get_data("forces")[0]
    force_data = [np.linalg.norm(x[0][1]) for x in force_data if len(x)!=0]
    
    force_data = [x for x in force_data if x<20]
    print(np.mean(force_data))
    #plt.plot(time_vector, velocity_data)
    plt.plot(force_data)
    plt.show()
    
res = simulation_rewarder.calculate_reward(simulation_output)
print('reward', res)
