
import optmizers_config
from obj_grasp.objects import get_obj_easy_box
from rule_sets import rule_extention

from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (
    ControlOptimizer)



# Init control optimization
cfg = optmizers_config.get_cfg_standart()
control_optimizer = ControlOptimizer(cfg)
cfg.get_rgab_object_callback = get_obj_easy_box
graph = rule_extention.get_three_finger()

# Run optimization
res = control_optimizer.start_optimisation(graph, is_debug=True)
print(res)

# Print result with visualisation
rew_func = control_optimizer.create_reward_function(graph)
rew_func(res[1], True)