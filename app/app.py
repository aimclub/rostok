import sys
import time
from pathlib import Path

import hyperparameters as hp
import mcts
from mcts_run_setup import config_with_standard

from rostok.graph_generators.mcts_helper import (make_mcts_step,
                                                 prepare_mcts_state_and_helper)
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
from rostok.library.rule_sets.ruleset_old_style import create_rules

# create rule vocabulary
rule_vocabul = create_rules()
# create blueprint for object to grasp
grasp_object_blueprint = get_object_parametrized_sphere(0.2, 1)
# create reward counter using run setup function
control_optimizer = config_with_standard(grasp_object_blueprint)
# Initialize MCTS
base_iteration_limit = hp.BASE_ITERATION_LIMIT
max_numbers_rules = hp.MAX_NUMBER_RULES
initial_graph = GraphGrammar()
graph_env = prepare_mcts_state_and_helper(initial_graph, rule_vocabul, control_optimizer,
                                          max_numbers_rules, Path("./results"))
mcts_helper = graph_env.helper
mcts_helper.report.non_terminal_rules_limit = max_numbers_rules
mcts_helper.report.search_parameter = base_iteration_limit

# the constant that determines how we reduce the number of iterations in the MCTS search
iteration_reduction_rate = hp.ITERATION_REDUCTION_TIME

start = time.time()
finish = False
n_steps = 0

while not finish:
    iteration_limit = base_iteration_limit - int(graph_env.counter_action / max_numbers_rules *
                                                 (base_iteration_limit * iteration_reduction_rate))
    searcher = mcts.mcts(iterationLimit=iteration_limit)
    finish, graph_env = make_mcts_step(searcher, graph_env, n_steps)
    n_steps += 1
    print(f"number iteration: {n_steps}, counter actions: {graph_env.counter_action} " +
          f"reward: {mcts_helper.report.get_best_info()[1]}")
ex = time.time() - start
print(f"time :{ex}")

report = mcts_helper.report
path = report.make_time_dependent_path()
report.save()
report.save_visuals()
report.save_lists()
report.save_means()

# additions to the file
with open(Path(path, "mcts_result.txt"), "a") as file:
    original_stdout = sys.stdout
    sys.stdout = file
    print()
    print("Object to grasp:", grasp_object_blueprint.shape)
    print("Object initial coordinats:", grasp_object_blueprint.pos)
    print("Time optimization:", ex)
    print("MAX_NUMBER_RULES:", hp.MAX_NUMBER_RULES)
    print("BASE_ITERATION_LIMIT:", hp.BASE_ITERATION_LIMIT)
    print("ITERATION_REDUCTION_TIME:", hp.ITERATION_REDUCTION_TIME)
    print("CRITERION_WEIGHTS:",
          [hp.TIME_CRITERION_WEIGHT, hp.FORCE_CRITERION_WEIGHT, hp.OBJECT_COG_CRITERION_WEIGHT])
    print("CONTROL_OPTIMIZATION_ITERATION:", hp.CONTROL_OPTIMIZATION_ITERATION)
    print("TIME_STEP_SIMULATION:", hp.TIME_STEP_SIMULATION)
    print("TIME_SIMULATION:", hp.TIME_SIMULATION)
    print("FLAG_TIME_NO_CONTACT:", hp.FLAG_TIME_NO_CONTACT)
    print("FLAG_TIME_SLIPOUT:", hp.FLAG_TIME_SLIPOUT)
    sys.stdout = original_stdout

simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_scenario
# visualisation in the end of the search
best_graph, reward, best_control = mcts_helper.report.get_best_info()
data = {"initial_value": best_control}
simulation_output = simulation_manager.run_simulation(best_graph, data, True)
res = -simulation_rewarder.calculate_reward(simulation_output)
print("Best reward obtained in the MCTS search:", res)