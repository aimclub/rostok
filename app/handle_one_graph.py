import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

import hyperparameters as hp
import mcts
from mcts_run_setup import config_with_standard

from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_ellipsoid, get_object_parametrized_sphere_smc
from rostok.library.rule_sets.ruleset_old_style_smc import create_rules
from rostok.library.rule_sets.simple_designs import get_one_finger_one_link, get_two_link_three_finger, get_one_finger_one_link, get_three_link_one_finger, get_three_same_link_one_finger, get_four_same_link_one_finger
from rostok.simulation_chrono.basic_simulation import SimulationResult
# create rule vocabulary
#rule_vocabul = create_rules()
# create blueprint for object to grasp
grasp_object_blueprint = get_object_parametrized_sphere_smc(0.05)
#grasp_object_blueprint = get_object_ellipsoid(10, 8, 14, 10)
# create reward counter using run setup function
# control_optimizer = config_with_standard_cable(grasp_object_blueprint)
# control_optimizer = config_with_standard_linear(grasp_object_blueprint)
control_optimizer = config_with_standard(grasp_object_blueprint)

path = Path("./app/single_graph/"+"graph_" + datetime.now().strftime("%yy_%mm_%dd_%HH_%MM")+".txt")
start = time.time()

ex = time.time() - start
print(f"time :{ex}")

# additions to the file
with open(path, "w") as file:
    original_stdout = sys.stdout
    sys.stdout = file
    print()
    print("Object to grasp:", grasp_object_blueprint.shape)
    dict = {item:getattr(hp, item) for item in dir(hp) if not item.startswith("__") and not item.endswith("__")}
    for key, value in dict.items():
        print(key, value)
    simulation_rewarder = control_optimizer.rewarder
    simulation_manager = control_optimizer.simulation_scenario
    # visualisation in the end of the search
    graph=get_three_link_one_finger()
    # graph=get_three_same_link_one_finger()
    #graph=get_four_same_link_one_finger()
    graph = get_one_finger_one_link()
    #graph=get_two_link_three_finger()
    #control = [10.5, 4.166667, 10.5, 10.5, 10.5, 10.5]
    #control = [1.05 , 1.683, 1.683, 0.417, 1.05 , 0.417]
    # graph=get_one_finger_one_link()
    control = [2]
    #control = [5]
    print('control:', control)
    data = control_optimizer.optim_parameters2data_control(control, graph)
    print(data)
    vis = True
    simulation_output: SimulationResult = simulation_manager.run_simulation(graph, data, [[-45]], vis, True)
    if not vis:
        fig = plt.figure(figsize=(12, 5))
        time_vector = simulation_output.time_vector
        velocity_data_idx = list(simulation_output.robot_final_ds.get_data("body_velocity").keys())
        trajectories = simulation_output.robot_final_ds.get_data("COG")[velocity_data_idx[-1]]
        velocity_data = simulation_output.robot_final_ds.get_data("body_velocity")[velocity_data_idx[-1]]
        velocity_data = [np.linalg.norm(x) for x in velocity_data]
        #velocity_data = [x[0] for x in velocity_data]
        plt.plot(time_vector, velocity_data)
        plt.show()

    res = simulation_rewarder.calculate_reward(simulation_output)
    print('reward', res)
    sys.stdout = original_stdout
