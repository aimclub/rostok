import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mcts
# imports from standard libs
import networkx as nx
# chrono imports
import pychrono as chrono
import rule_extention
from control_optimisation import (create_grab_criterion_fun, create_traj_fun,
                                  get_object_to_grasp)

from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact,
                                                FlagSlipout)
from rostok.graph_generators.mcts_helper import (make_mcts_step,
                                                 prepare_mcts_state_and_helper)
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (
    ConfigRewardFunction, ControlOptimizer)


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()


# %% Create extension rule vocabulary
rule_vocabul, node_features = rule_extention.init_extension_rules()

# %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [3, 1, 1, 2]

cfg = ConfigRewardFunction()
cfg.bound = (2, 10)
cfg.iters = 5
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.005
cfg.time_sim = 2
cfg.flags = [FlagMaxTime(2), FlagNotContact(1), FlagSlipout(0.5, 0.5)]
"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)
# %% Init mcts parameters

# Hyperparameters mctss
base_iteration_limit = 50

# Initialize MCTS
#searcher = mcts.mcts(iterationLimit=iteration_limit)
finish = False

initial_graph = GraphGrammar()
max_numbers_rules = 20
# Create graph environments for algorithm (not gym)
graph_env = prepare_mcts_state_and_helper(initial_graph, rule_vocabul, control_optimizer, max_numbers_rules,
                                          Path("./results"))
mcts_helper = graph_env.helper
mcts_helper.report.non_terminal_rules_limit = max_numbers_rules
mcts_helper.report.search_parameter = base_iteration_limit
n_steps = 0
#%% Run first algorithm
start = time.time()
while not finish:
    iteration_limit = base_iteration_limit - int(graph_env.counter_action/max_numbers_rules * (base_iteration_limit*0.7))
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
    gb_params = get_object_to_grasp().kwargs
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
#report.plot_means()
#report.draw_best_graph()
