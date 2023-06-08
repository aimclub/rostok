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

def get_leaderboard(history, top_n: int = 10) -> str:
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

def plot_mean_reward(history: OptHistory):
    means = list(map(np.mean, history.historical_fitness))
    to_positive = [-1 * i for i in means][:-1]
    
    
    plt.plot(to_positive)
    plt.xlabel("Population")
    plt.ylabel("Mean reward")
    #plt.savefig(prefix + "means.svg")
    #plt.savefig(prefix + "means.fig")

def plot_mean_reward_3(history: OptHistory):
    sort = list(map(sorted, history.historical_fitness))
    to_positive = []
    for i in sort:
        to_positive.append(np.mean(i[0:3]))
    
    to_positive = [-1 * i for i in to_positive][:-1]
    final = []

    
    plt.plot(to_positive)
    plt.xlabel("Population")
    plt.ylabel("Mean reward 3")
    #plt.savefig(prefix + "means.svg")
    #plt.savefig(prefix + "means.fig")
    
    
 
reports_label = ["cylinder", "box", "ellipsoid"]
history1 : OptHistory = pickle.load( open( "1686240148get_object_parametrized_cylinder_21_589", "rb" ) )
history2 : OptHistory = pickle.load( open( "1686238049get_object_easy_box_8_827", "rb" ) )
history3 : OptHistory = pickle.load( open( "1686240116get_obj_hard_get_obj_hard_large_ellipsoid_6_919", "rb" ) )
plt.figure()
plot_mean_reward(history1)
plot_mean_reward(history2)
plot_mean_reward(history3)
plt.ylim((2, 11))
plt.legend(reports_label)
plt.show()