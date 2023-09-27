import pickle
from mcts_run_setup import config_with_standard_graph
from rostok.adapters.golem_adapter import GraphGrammarAdapter
from golem.core.optimisers.optimizer import OptHistory
import numpy as np
import matplotlib.pyplot as plt
from rostok.library.rule_sets import ruleset_old_style_graph
from rostok.library.obj_grasp.objects import get_object_box
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
    plt.ylabel("Median reward")
    plt.savefig(prefix + "median.svg")
    #plt.savefig(prefix + "median.fig")

 
 
obj = get_object_box(1.2, 0.5, 0.8, 0)
rules, torque_dict = ruleset_old_style_graph.create_rules()
name = "1695740607mock_with_build_mech-11_221"
history : OptHistory = pickle.load( open( name, "rb" ) )
adapter_local = GraphGrammarAdapter()
optic = config_with_standard_graph(obj, torque_dict)
scena = optic.simulation_scenario

ebaca = get_leaderboard(history)
ebaca_restored = list(map(adapter_local.restore, ebaca))
plot_mean_reward(history, name)
plot_median_reward(history, name)

for i in ebaca_restored:
    controll_parameters = optic.build_control_from_graph(i)
    controll_parameters = {"initial_value": controll_parameters}
    scena.run_simulation(i, controll_parameters, vis=True)
 
 