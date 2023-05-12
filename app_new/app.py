import sys
import time
import matplotlib.pyplot as plt
import mcts
import numpy as np
from pathlib import Path
from rostok.simulation_chrono.basic_simulation import ConstTorqueGrasp
from rostok.criterion.simulation_flags import FlagContactTimeOut, FlagFlyingApart, FlagSlipout
from simple_designs import get_three_link_one_finger_with_no_control, get_two_link_one_finger
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from control_optimisation import get_object_to_grasp
from rostok.criterion.criterion_calc import SimulationReward,TimeCriterion, ForceCriterion, ObjectCOGCriterion
from rostok.trajectory_optimizer.control_optimizer import CounterWithOptimization
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_generators.mcts_helper import (make_mcts_step, prepare_mcts_state_and_helper)
from rostok.library.rule_sets.ruleset_old_style_graph_nonails import create_rules


rule_vocabul, _ = create_rules()
# construct a simulation manager
simulation_control = ConstTorqueGrasp(0.005, 3)
# add object to grasp
simulation_control.grasp_object_callback = get_object_to_grasp
# create flags
simulation_control.add_flag(FlagContactTimeOut(2))
simulation_control.add_flag(FlagFlyingApart(10))
simulation_control.add_flag(FlagSlipout(1.5))
#create criterion manager
simulation_rewarder = SimulationReward()
#create criterions and add them to manager
simulation_rewarder.add_criterion(TimeCriterion(),10.0)
simulation_rewarder.add_criterion(ForceCriterion(),5.0)
simulation_rewarder.add_criterion(ObjectCOGCriterion(),2.0)
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
