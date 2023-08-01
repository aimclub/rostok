import sys

sys.path.append("./examples/")
import time
from pathlib import Path

import hyperparameters as hp
import mcts
from mcts_run_setup import config_with_standard_multiobject

from rostok.graph_generators.mcts_helper import (make_mcts_step, prepare_mcts_state_and_helper, CheckpointMCTS)
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_cylinder, get_object_box
from rostok.library.rule_sets.ruleset_new_style import create_rules

weights = [1, 1, 1]
# create rule vocabulary
rule_vocabul = create_rules()
# create blueprint for object to grasp
grasp_object_blueprint = []
grasp_object_blueprint.append(get_object_parametrized_sphere(0.4))
grasp_object_blueprint.append(get_object_cylinder(0.4, 0.4, 0))
grasp_object_blueprint.append(get_object_box(0.5,0.2,0.7,0))
# create reward counter using run setup function
control_optimizer = config_with_standard_multiobject(grasp_object_blueprint, weights)
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
checkpointer = CheckpointMCTS(mcts_helper.report, "App", rewrite=True)
start = time.time()
finish = False
n_steps = 0

while not finish:
    iteration_limit = base_iteration_limit - int(graph_env.counter_action / max_numbers_rules *
                                                 (base_iteration_limit * iteration_reduction_rate))
    searcher = mcts.mcts(iterationLimit=iteration_limit)
    finish, graph_env = make_mcts_step(searcher, graph_env, n_steps, checkpointer)
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
    for k, (obj, weight) in enumerate(zip(grasp_object_blueprint, weights)):
        print(f"Objects to grasp {k}:", obj.shape)
        print(f"Object initial coordinats {k}:", obj.pos)
        print(f"Object weight {k}:", weight)
    print("Time optimization:", ex)
    dict = {item:getattr(hp, item) for item in dir(hp) if not item.startswith("__") and not item.endswith("__")}
    for key, value in dict.items():
        print(key, "=", value)
    sys.stdout = original_stdout

simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_scenario[0]
# visualisation in the end of the search
best_graph, reward, best_control = mcts_helper.report.get_best_info()
data = control_optimizer.optim_parameters2data_control(best_control)
simulation_output = simulation_manager[0].run_simulation(best_graph, data[0], True)
res = simulation_rewarder.calculate_reward(simulation_output)
print("Best reward obtained in the MCTS search:", res)