import sys
import time
from pathlib import Path

import hyperparameters as hp
import mcts
from mcts_run_setup import config_with_standard_graph

from rostok.graph_generators.mcts_helper import (make_mcts_step, prepare_mcts_state_and_helper,
                                                 CheckpointMCTS)
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_parametrized_tilt_ellipsoid
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules

grasp_object_blueprint = get_object_parametrized_tilt_ellipsoid(1, 0.8, 1.4, 10)

checkpointer, graph_env, report, __, __ = CheckpointMCTS.restore_optimization(
    "AppGraphEllipsoid_3", 1, grasp_object_blueprint)

base_iteration_limit = report.search_parameter
max_numbers_rules = report.non_terminal_rules_limit
iteration_reduction_rate = hp.ITERATION_REDUCTION_TIME

start = time.time()
finish = False
n_steps = graph_env.helper.step_counter
while not finish:
    iteration_limit = base_iteration_limit - int(graph_env.counter_action / max_numbers_rules *
                                                 (base_iteration_limit * iteration_reduction_rate))
    searcher = mcts.mcts(iterationLimit=iteration_limit)
    finish, graph_env = make_mcts_step(searcher, graph_env, n_steps, checkpointer)
    n_steps += 1
    print(f"number iteration: {n_steps}, counter actions: {graph_env.counter_action} " +
          f"reward: {report.get_best_info()[1]}")
ex = time.time() - start
print(f"time :{ex}")
# saving results of the search
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

simulation_rewarder = graph_env.optimizer.rewarder
simulation_manager = graph_env.optimizer.simulation_scenario
# visualisation in the end of the search
best_graph, reward, best_control = graph_env.helper.report.get_best_info()
data = {"initial_value": best_control}
simulation_output = simulation_manager.run_simulation(best_graph, data, True)
res = -simulation_rewarder.calculate_reward(simulation_output)
print("Best reward obtained in the MCTS search:", res)