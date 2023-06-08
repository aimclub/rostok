import pickle
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_easy_box, get_obj_hard_get_obj_hard_large_ellipsoid
from mcts_run_setup import config_with_standard_graph
from rostok.adapters.golem_adapter import (GraphGrammarAdapter)

import pickle
from golem.core.optimisers.optimizer import OptHistory
from rostok.library.rule_sets.rule_extention_merge import rule_vocab, torque_dict
import numpy as np
import matplotlib.pyplot as plt
from rostok.library.obj_grasp.objects import get_object_parametrized_cylinder, get_object_easy_box, get_obj_hard_get_obj_hard_large_ellipsoid
import networkx as nx

obj_ell = get_obj_hard_get_obj_hard_large_ellipsoid()
grasp_object_cyl = get_object_parametrized_cylinder(0.4, 1, 0.7)

def get_leaderboard(history, top_n: int = 5) -> str:
    """
    Prints ordered description of the best solutions in history
    :param top_n: number of solutions to print
    """
    # Take only the first graph's appearance in history
    individuals_with_positions \
        = list({ind.graph.descriptive_id: (ind, gen_num, ind_num)
                for gen_num, gen in enumerate(history.individuals)
                for ind_num, ind in reversed(list(enumerate(gen)))}.values())
    #seta = set(individuals_with_positions)
    top_individuals = sorted(individuals_with_positions,
                            key=lambda pos_ind: pos_ind[0].fitness, reverse=True)[:top_n]
    list_graph = []
    for ind in top_individuals:
        list_graph.append(ind[0].graph)
    return list_graph

def plot_mean_reward(history: OptHistory, prefix: str):
    means = list(map(np.mean, history.historical_fitness))
    to_positive = [-1 * i for i in means]
    plt.figure()
    
    plt.plot(to_positive)
    plt.xlabel("Population")
    plt.ylabel("Mean reward")
    plt.savefig(prefix + "means.svg")
    #plt.savefig(prefix + "means.fig")
    
def plot_median_reward(history: OptHistory, prefix: str):
    median = list(map(np.median, history.historical_fitness))
    to_positive = [-1 * i for i in median]
    plt.figure()
    plt.plot(to_positive)
    plt.xlabel("Population")
    plt.ylabel("Mean reward")
    plt.savefig(prefix + "median.svg")
    #plt.savefig(prefix + "median.fig")


obj = grasp_object_cyl
 
control_optimizer = config_with_standard_graph(obj, torque_dict)

simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_control

history : OptHistory = pickle.load( open( "1686248648get_object_parametrized_cylinder_17_523", "rb" ) )
adapter_local = GraphGrammarAdapter()

ebaca = get_leaderboard(history)
ebaca_restored = list(map(adapter_local.restore, ebaca))



def build_wrapperd_r(graph):
    res, _ = control_optimizer.count_reward(graph)
    return -res

for i in ebaca_restored:
    res = build_wrapperd_r(i)
    print(-res)

