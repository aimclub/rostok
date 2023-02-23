import sys
import time
from copy import deepcopy
from pathlib import Path

import mcts
import optmizers_config
from obj_grasp.objects import get_obj_easy_box, get_obj_hard_ellipsoid
from rule_sets import rule_extention_graph

import hyperparameters as hp
from rostok.graph_generators.mcts_helper import (make_mcts_step,
                                                 prepare_mcts_state_and_helper)
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (
    ControlOptimizer)
from rule_sets.ruleset_old_style_graph import create_rules


rule_vocabul, torque_dict = create_rules()
#rule_vocabul = deepcopy(rule_extention_graph.rule_vocab)
cfg = optmizers_config.get_cfg_graph(torque_dict)
#cfg.get_rgab_object_callback = get_obj_hard_ellipsoid
cfg.get_rgab_object_callback = get_obj_easy_box
control_optimizer = ControlOptimizer(cfg)
 
base_iteration_limit = hp.BASE_ITERATION_LIMIT
max_numbers_rules = hp.MAX_NUMBER_RULES
iteration_reduction_rate = hp.ITERATION_REDUCTION_TIME

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
    print("Time optimization:", ex)
    print("MAX_NUMBER_RULES:", hp.MAX_NUMBER_RULES)
    print("BASE_ITERATION_LIMIT:", hp.BASE_ITERATION_LIMIT)
    print("ITERATION_REDUCTION_TIME:", hp.ITERATION_REDUCTION_TIME)
    print("CRITERION_WEIGHTS:", hp.CRITERION_WEIGHTS)
    print("CONTROL_OPTIMIZATION_ITERATION:", hp.CONTROL_OPTIMIZATION_ITERATION)
    print("TIME_STEP_SIMULATION:", hp.TIME_STEP_SIMULATION)
    print("TIME_SIMULATION:", hp.TIME_SIMULATION)
    print("FLAG_TIME_NO_CONTACT:", hp.FLAG_TIME_NO_CONTACT)
    print("FLAG_TIME_SLIPOUT:", hp.FLAG_TIME_SLIPOUT)
    sys.stdout = original_stdout   

# visualisation in the end of the search
best_graph, reward, best_control = mcts_helper.report.get_best_info()
func_reward = control_optimizer.create_reward_function(best_graph)
res = -func_reward(best_control, True)
print("Best reward obtained in the MCTS search:", res)
