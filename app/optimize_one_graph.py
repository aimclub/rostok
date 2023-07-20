import sys
import time
from pathlib import Path
from datetime import datetime

import hyperparameters as hp
import mcts
from mcts_run_setup import config_with_standard, config_with_standard_tendon

from rostok.graph_generators.mcts_helper import (make_mcts_step, prepare_mcts_state_and_helper,
                                                 CheckpointMCTS)
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_parametrized_tilt_ellipsoid
from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.library.rule_sets.simple_designs import (get_three_link_one_finger, 
                                                     get_three_same_link_one_finger, 
                                                     get_four_same_link_one_finger, 
                                                     get_two_link_three_finger, 
                                                     get_three_link_three_finger, 
                                                     get_three_link_three_finger_scale,
                                                     get_three_link_three_finger_scale_dist)

# create rule vocabulary
rule_vocabul = create_rules()
# create blueprint for object to grasp
#grasp_object_blueprint = get_object_parametrized_sphere(0.5, 1)
grasp_object_blueprint = get_object_parametrized_tilt_ellipsoid(1, 0.8, 1.4, 10)
# create reward counter using run setup function
#control_optimizer = config_with_standard_tendon(grasp_object_blueprint)
control_optimizer = config_with_standard(grasp_object_blueprint)
path = Path("./app/single_graph/" + "graph_" + datetime.now().strftime("%yy_%mm_%dd_%HH_%MM") +
            ".txt")
start = time.time()

# additions to the file
with open(path, "w") as file:
    original_stdout = sys.stdout
    sys.stdout = file
    print()
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
    control_optimizer.limit = 20
    # visualisation in the end of the search
    graph = get_three_link_one_finger()
    graph = get_three_same_link_one_finger()
    graph = get_two_link_three_finger()
    graph = get_three_link_three_finger_scale()
    #graph = get_three_link_three_finger_scale_dist()
    #graph = get_three_link_three_finger()
    reward, control = control_optimizer.calculate_reward(graph)
    print('control:', control)
    data = control_optimizer.optim_parameters2data_control(control, graph)
    print(data)
    simulation_output = simulation_manager.run_simulation(graph, data, True, True)
    res = simulation_rewarder.calculate_reward(simulation_output)
    print('reward', res)
    ex = time.time() - start
    print(f"time :{ex}")
    sys.stdout = original_stdout
