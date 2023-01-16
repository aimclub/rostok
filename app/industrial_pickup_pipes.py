from time import sleep
import numpy as np
import mcts
import matplotlib.pyplot as plt

# imports from standard libs
import networkx as nx
# chrono imports
import pychrono as chrono

from control_optimisation import create_grab_criterion_fun, create_traj_fun
import rostok.intexp as intexp
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagSlipout, FlagNotContact,
                                                FlagFlyingApart)
from rostok.utils.result_saver import MCTSReporter, load_reporter
import rostok.graph_generators.graph_environment as env

import rule_grasp_pipe
import pickup_pipes_utils as ppu

PATH_TO_PIPE_OBJ = './examples/models/custom/pipe_mul_10.obj'
PATH_TO_PIPE_XML = './examples/models/custom/pipe.xml'

# # %% Create extension rule vocabulary

rule_vocabul, node_features = rule_grasp_pipe.create_rules_to_pickup_pipe(PATH_TO_PIPE_OBJ, PATH_TO_PIPE_XML)

# # %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [3, 1, 1, 2]

max_time = 1
cfg = ConfigRewardFunction()
cfg.bound = (1, 2)
cfg.iters = 2
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.0005
cfg.time_sim = max_time
cfg.flags = [
    FlagMaxTime(max_time)]#,
    # FlagNotContact(max_time / 4 - 0.2),
    # FlagSlipout(max_time / 4 + 0.2, 0.2),
    # FlagFlyingApart(4)
# ]

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = ppu.create_builder_grab_object(PATH_TO_PIPE_OBJ,PATH_TO_PIPE_XML)
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)

# # %% Init mcts parameters

# Hyperparameters mctss
iteration_limit = 3

# Initialize MCTScl
searcher = mcts.mcts(iterationLimit=iteration_limit)
finish = False

G = GraphGrammar()
max_numbers_rules = 5*3+1
# Create graph envirenments for algorithm (not gym)
graph_env = env.GraphVocabularyEnvironment(G, rule_vocabul, max_numbers_rules)

graph_env.set_control_optimizer(control_optimizer)

reporter = MCTSReporter.get_instance()
reporter.rule_vocabulary = rule_vocabul
reporter.initialize()

# #%% Run first algorithm
iter = 0
while not finish:
    action = searcher.search(initialState=graph_env)
    finish, final_graph, opt_trajectory, path = graph_env.step(action, False)
    iter += 1
    print(f"number iteration: {iter}, counter actions: {graph_env.counter_action}")

# path = reporter.dump_results()
# reporter = load_reporter('results\MCTS_report_22y_12m_30d_03H_30M')
# best_graph, reward, best_control = reporter.get_best_info()
# # best_control = [float(x) for x in best_control]
# func_reward = control_optimizer.create_reward_function_pickup(best_graph)
# res = -func_reward(best_control)
# print(res)