import sys
import time
from copy import deepcopy
from pathlib import Path

import mcts
import optmizers_config
from obj_grasp.objects import get_obj_easy_box
from rule_sets import rule_extention

from rostok.graph_generators.mcts_helper import (make_mcts_step,
                                                 prepare_mcts_state_and_helper)
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (
    ControlOptimizer)

rule_vocabul = deepcopy(rule_extention.rule_vocab)
cfg = optmizers_config.get_cfg_standart()
cfg.get_rgab_object_callback = get_obj_easy_box
control_optimizer = ControlOptimizer(cfg)
 
base_iteration_limit = 50
max_numbers_rules = 20
iteration_reduction_rate = 0.7

# Create graph environments for algorithm (not gym)
graph_env = prepare_mcts_state_and_helper(GraphGrammar(), rule_vocabul, control_optimizer, max_numbers_rules,
                                          Path("./results"))
mcts_helper = graph_env.helper
mcts_helper.report.non_terminal_rules_limit = max_numbers_rules
mcts_helper.report.search_parameter = base_iteration_limit



start = time.time()
finish = False
n_steps = 0

while not finish:
    iteration_limit = base_iteration_limit - int(graph_env.counter_action/max_numbers_rules * (base_iteration_limit*iteration_reduction_rate))
    searcher = mcts.mcts(iterationLimit=iteration_limit)
    finish, graph_env = make_mcts_step(searcher, graph_env, n_steps)
    n_steps += 1
    print(f"number iteration: {n_steps}, counter actions: {graph_env.counter_action} " +
          f"reward: {mcts_helper.report.get_best_info()[1]}")
ex = time.time() - start
print(f"time :{ex}")
# saving results of the search
report = mcts_helper.report
path = report.make_time_dependent_path()
report.save()
report.save_visuals()
report.save_lists()
report.save_means()


# additions to the file
with open(Path(path, "mcts_result.txt"), "a") as file:
    gb_params = cfg.get_rgab_object_callback().kwargs
    original_stdout = sys.stdout
    sys.stdout = file
    print()
    print("Object to grasp:", gb_params.get("shape"))
    print("Object initial coordinats:", gb_params.get("pos"))
    sys.stdout = original_stdout   

# visualisation in the end of the search
best_graph, reward, best_control = mcts_helper.report.get_best_info()
func_reward = control_optimizer.create_reward_function(best_graph)
res = -func_reward(best_control, True)
print("Best reward obtained in the MCTS search:", res)
