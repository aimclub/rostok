from pathlib import Path

from rostok.utils.pickle_save import load_saveable



#report.plot_means()

from pathlib import Path
import time
import matplotlib.pyplot as plt
import mcts
# imports from standard libs
import networkx as nx
# chrono imports
import pychrono as chrono
import optmizers_config
from obj_grasp.objects import get_obj_easy_box, get_obj_hard_ellipsoid
from rostok.graph_generators.mcts_helper import prepare_mcts_state_and_helper, make_mcts_step
from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact, FlagSlipout)
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (ConfigRewardFunction, ControlOptimizer)
from rule_sets.ruleset_old_style_graph import create_rules

def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()

report = load_saveable(Path(r".\results\Reports_23y_02m_21d_21H_13M\MCTS_data.pickle"))
# %% Create extension rule vocabulary
rule_vocabul, torque_dict = create_rules()
#rule_vocabul = deepcopy(rule_extention_graph.rule_vocab)
cfg = optmizers_config.get_cfg_graph(torque_dict)
#cfg.get_rgab_object_callback = get_obj_hard_ellipsoid
cfg.get_rgab_object_callback = get_obj_easy_box
control_optimizer = ControlOptimizer(cfg)

best_graph, reward, best_control = report.get_best_info()
func_reward = control_optimizer.create_reward_function(best_graph)
res = -func_reward(best_control, True)
