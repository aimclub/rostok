from control_optimisation import create_grab_criterion_fun, create_traj_fun, get_object_to_grasp
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
import rule_extention
import mcts
import rostok.graph_generators.graph_environment as env
import matplotlib.pyplot as plt

# imports from standard libs
import networkx as nx

# chrono imports
import pychrono as chrono



def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(graph, dim=2), node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()

# %% Create extension rule vocabulary

rule_vocabul, node_features = rule_extention.init_extension_rules()

# %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [1, 1, 1, 1]

cfg = ConfigRewardFunction()
cfg.bound = (0, 10)
cfg.iters = 10
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.0015
cfg.time_sim = 3
cfg.flags = [FlagMaxTime(3), FlagNotContact(1), FlagSlipout(1, 0.5)]

"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)

# %% Init mcts parameters

G = GraphGrammar()
max_numbers_rules = 5

# Hyperparameters mcts
iteration_limit = 10

# Initilize MCTS
searcher = mcts.mcts(iterationLimit=iteration_limit)
finish = False

G = GraphGrammar()
max_numbers_rules = 7
# Create graph envirenments for algorithm (not gym)
graph_env = env.GraphVocabularyEnvironment(G, rule_vocabul, max_numbers_rules)

graph_env.set_control_optimizer(control_optimizer)

# %% Run first algorithm
iter = 0
while not finish:
    action = searcher.search(initialState=graph_env)
    finish, final_graph, opt_trajectory = graph_env.step(action,False)
    iter +=1
    print(f"number iteration: {iter}, counter actions: {graph_env.counter_action}, reward: {graph_env.reward}")
    
# %%
 

func_reward = control_optimizer.create_reward_function(final_graph)
res = func_reward(opt_trajectory, True)
print(res)