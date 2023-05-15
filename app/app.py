import time
import sys
from pathlib import Path
import hyperparameters as hp
import matplotlib.pyplot as plt
import mcts
import numpy as np
from simple_designs import (get_three_link_one_finger_with_no_control,
                            get_two_link_one_finger)

from rostok.criterion.criterion_calculation import (ForceCriterion,
                                                    ObjectCOGCriterion,
                                                    SimulationReward,
                                                    TimeCriterion)
from rostok.criterion.simulation_flags import (FlagContactTimeOut,
                                               FlagFlyingApart, FlagSlipout)
from rostok.graph_generators.mcts_helper import (make_mcts_step,
                                                 prepare_mcts_state_and_helper)
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.simulation_chrono.simulation_scenario import ConstTorqueGrasp
from rostok.trajectory_optimizer.control_optimizer import CounterWithOptimization
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator


rule_vocabul = create_rules()
# construct a simulation manager
simulation_control = ConstTorqueGrasp(0.005, 3)
# add object to grasp
grasp_object_blueprint = get_object_parametrized_sphere(0.2, 1)
simulation_control.grasp_object_callback = lambda :creator.create_environment_body(grasp_object_blueprint)
# create flags
simulation_control.add_flag(FlagContactTimeOut(2))
simulation_control.add_flag(FlagFlyingApart(10))
simulation_control.add_flag(FlagSlipout(1.5))
#create criterion manager
simulation_rewarder = SimulationReward()
#create criterions and add them to manager
simulation_rewarder.add_criterion(TimeCriterion(3), 10.0)
simulation_rewarder.add_criterion(ForceCriterion(), 5.0)
simulation_rewarder.add_criterion(ObjectCOGCriterion(), 2.0)
#create optimization manager
control_optimizer = CounterWithOptimization(simulation_control, simulation_rewarder)

# graph = get_two_link_one_finger()

# print(optimizer.count_reward(graph))

# print(get_joint_vector_from_graph(graph))
# control = np.random.random(len(get_joint_vector_from_graph(graph)))
# print(control)
#data = {"initial_value": list(control)}
#simulation_control.run_simulation(graph, data)

# Hyperparameters mcts
base_iteration_limit = 50

# Initialize MCTS
finish = False

initial_graph = GraphGrammar()
max_numbers_rules = 20
# Create graph environments for algorithm (not gym)
graph_env = prepare_mcts_state_and_helper(initial_graph, rule_vocabul, control_optimizer,
                                          max_numbers_rules, Path("./results"))
mcts_helper = graph_env.helper
mcts_helper.report.non_terminal_rules_limit = max_numbers_rules
mcts_helper.report.search_parameter = base_iteration_limit
n_steps = 0
start = time.time()
# the constant that determines how we reduce the number of iterations in the MCTS search
iteration_reduction_rate = 0.7
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
    gb_params = grasp_object_blueprint.kwargs
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