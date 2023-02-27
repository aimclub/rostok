import datetime
from copy import deepcopy
from functools import partial
import pickle
import time
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import \
    CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import \
    GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import (
    MutationStrengthEnum, MutationTypesEnum)
from golem.core.optimisers.genetic.operators.regularization import \
    RegularizationTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from obj_grasp.objects import get_obj_hard_ellipsoid, get_object_to_grasp_sphere, get_obj_hard_ellipsoid_move
from optmizers_config import get_cfg_graph
from rule_sets.ruleset_old_style_graph import create_rules

from rostok.adapters.golem_adapter import (GraphGrammarAdapter,
                                           GraphGrammarFactory)
from rostok.graph_grammar.graph_utils import plot_graph
from rostok.graph_grammar.make_random_graph import make_random_graph
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
import random
from mutation_logik import add_mut, del_mut
from golem.core.optimisers.genetic.operators import crossover
import pickle
from golem.core.optimisers.optimizer import OptHistory
from rule_sets.rule_extention_golem_edition import rule_vocab, torque_dict
import numpy as np
import matplotlib.pyplot as plt
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


history : OptHistory = pickle.load( open( "1677432514get_obj_hard_ellipsoid_move_31_3387", "rb" ) )
adapter_local = GraphGrammarAdapter()
cfg = get_cfg_graph(torque_dict)
cfg.get_rgab_object_callback = get_obj_hard_ellipsoid_move
optic = ControlOptimizer(cfg)

ebaca = get_leaderboard(history)
ebaca_restored = list(map(adapter_local.restore, ebaca))
plot_mean_reward(history, "1677432514get_obj_hard_ellipsoid_move_31_3387")
plot_median_reward(history, "1677432514get_obj_hard_ellipsoid_move_31_3387")

for i in ebaca_restored:
    rews  = optic.create_reward_function(i, True)
    resik = rews([0])
    print(resik)
