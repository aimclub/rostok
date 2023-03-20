
import optmizers_config
from rostok.library.obj_grasp.objects import get_obj_easy_box, get_obj_hard_mesh_bukvg, get_obj_hard_mesh_mikki, get_obj_hard_mesh_piramida
from rostok.library.rule_sets import rule_extention

from rostok.graph_grammar.graph_grammar import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (
    ControlOptimizer)



# Init control optimization
cfg = optmizers_config.get_cfg_standart()
cfg.gravity_vector = [0, 9.8, 0]
cfg.time_saturation_gravity = 0.5
cfg.time_start_gravity = 1
control_optimizer = ControlOptimizer(cfg)
cfg.get_rgab_object_callback = get_obj_hard_mesh_mikki
graph = rule_extention.get_three_finger()

# Run optimization
res = control_optimizer.start_optimisation(graph, is_debug=True)
print(res)

# Print result with visualisation
rew_func = control_optimizer.create_reward_function(graph)
rew_func(res[1], True)