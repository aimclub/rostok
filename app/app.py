import matplotlib.pyplot as plt
import mcts
# imports from standard libs
import networkx as nx
# chrono imports
import pychrono as chrono
import rule_extention
from control_optimisation import (create_grab_criterion_fun, create_traj_fun,
                                  get_object_to_grasp)

import rostok.graph_generators.graph_environment as env
from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact,
                                                FlagSlipout)
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (
    ConfigRewardFunction, ControlOptimizer)
#from rostok.utils.result_saver import read_report
from rostok.utils.result_saver import MCTSReporter


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    #plt.savefig("./results/graph.jpg")
    plt.show()


# %% Create extension rule vocabulary

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
iteration_limit = 2

# Initialize MCTS
searcher = mcts.mcts(iterationLimit=iteration_limit)
finish = False

G = GraphGrammar()
max_numbers_rules = 2
# Create graph envirenments for algorithm (not gym)
graph_env = env.GraphVocabularyEnvironment(G, rule_vocabul, max_numbers_rules)

graph_env.set_control_optimizer(control_optimizer)

reporter = MCTSReporter.get_instance()
reporter.rule_vocabulary = rule_vocabul
reporter.initialize()

#%% Run first algorithm
iter = 0
while not finish:
    action = searcher.search(initialState=graph_env)
    finish, final_graph = graph_env.step(action, False)
    iter += 1
    print(
        f"number iteration: {iter}, counter actions: {graph_env.counter_action}, reward: {reporter.get_best_info()[1]}"
    )


path = reporter.dump_results()
best_graph,  reward, best_control = reporter.get_best_info()
# best_control = [float(x) for x in best_control]
func_reward = control_optimizer.create_reward_function(best_graph)
res = - func_reward(best_control)
plot_graph(best_graph)
print(res)
