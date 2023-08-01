import sys
import time
import pickle
from pathlib import Path

import hyperparameters as hp
import mcts
from mcts_run_setup import config_with_standard_graph

from rostok.graph_generators.mcts_helper import (make_mcts_step, prepare_mcts_state_and_helper,
                                                 CheckpointMCTS)
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules

grasp_object_blueprint = get_object_parametrized_sphere(0.5)

checkpointer, graph_env = CheckpointMCTS.restore_optimization(
    "App", 1, grasp_object_blueprint)

report = graph_env.helper.report
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
with open(Path(path, "object.pickle"), "wb") as file:
    pickle.dump(grasp_object_blueprint, file)
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
    dict = {
        item: getattr(hp, item)
        for item in dir(hp)
        if not item.startswith("__") and not item.endswith("__")
    }
    for key, value in dict.items():
        print(key, "=", value)
    sys.stdout = original_stdout

simulation_rewarder = graph_env.optimizer.rewarder
simulation_manager = graph_env.optimizer.simulation_scenario
# visualisation in the end of the search
best_graph, reward, best_control = graph_env.helper.report.get_best_info()
data = {"initial_value": best_control}
simulation_output = simulation_manager.run_simulation(best_graph, data, True)
res = -simulation_rewarder.calculate_reward(simulation_output)
print("Best reward obtained in the MCTS search:", res)