from mcts_run_setup import config_tendon

from rostok.library.obj_grasp.objects import get_object_sphere
from rostok.library.rule_sets.simple_designs import (
    get_three_link_one_finger, get_three_link_one_finger_independent)

# create blueprint for object to grasp
grasp_object_blueprint = get_object_sphere(0.05)

# create reward counter using run setup function
# control_optimizer = config_with_const_troques(grasp_object_blueprint)
 
control_optimizer = config_tendon(grasp_object_blueprint)

graph = get_three_link_one_finger_independent()
graph = get_three_link_one_finger()

control_optimizer.calculate_reward(graph)