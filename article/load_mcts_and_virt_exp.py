from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import mcts

# imports from standard libs
import networkx as nx
import optmizers_config
# chrono imports
import pychrono as chrono
from obj_grasp.objects import get_obj_easy_box, get_obj_hard_ellipsoid
#from rule_sets import rule_extention_graph
from rule_sets.ruleset_old_style_graph import create_rules

from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact,
                                                FlagSlipout)
from rostok.graph_generators.mcts_helper import (make_mcts_step,
                                                 prepare_mcts_state_and_helper)
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from rostok.utils.pickle_save import load_saveable


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()


report = load_saveable(Path("/home/human/rostok-team/rostok/results/Reports_23y_02m_24d_13H_27M/MCTS_data.pickle"))
# %% Create extension rule vocabulary
rule_vocabul, torque_dict = create_rules()
cfg = optmizers_config.get_cfg_graph(torque_dict)

cfg.get_rgab_object_callback = get_obj_hard_ellipsoid
control_optimizer = ControlOptimizer(cfg)
seen_graphs = deepcopy(report.seen_graphs.graph_list)
key_sort = lambda x: x.reward
seen_graphs.sort(key=key_sort) 
for num ,graph_and_res in enumerate(reversed(seen_graphs)):
    if num > 10:
        break
    rewa = control_optimizer.create_reward_function(graph_and_res.graph)
    rewa(graph_and_res.control, True)
    #print(graph_and_res.reward)