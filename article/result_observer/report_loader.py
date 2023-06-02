from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import mcts
# imports from standard libs
import networkx as nx
import numpy as np

import configurater as c

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



# %% Create extension rule vocabulary

def one_best_render(report, control_optimizer):
    best_graph, reward, best_control = report.get_best_info()
    #best_graph, reward, best_control = report.get_main_info()
    func_reward = control_optimizer.create_reward_function(best_graph)
    plot_graph(best_graph)
    best_control = []
    res = -func_reward(best_control, True)
    print(f"Save reward {reward}, Current reward {res}")

def top_ten_best_render(report, control_optimizer, top_number):
    seen_graphs = deepcopy(report.seen_graphs.graph_list)
    key_sort = lambda x: -x.reward
    seen_graphs.sort(key=key_sort) 
    for num ,graph_and_res in enumerate(seen_graphs):
        if top_number[0]-1 > num:
            continue
        if num >= top_number[1]:
            break
        rewa = control_optimizer.create_reward_function(graph_and_res.graph)
        rewa(graph_and_res.control, True)
        #print(graph_and_res.reward)

def save_svg_mean_reward(reporter, legend, name, filter = False):
    path = "./article/result_observer/figure/" + name + ".svg"
    arr_mean_rewards = []

    rewards = []
    for state in reporter[0].seen_states.state_list:
        i = state.step
        if len(rewards) == i:
            rewards.append([state.reward])
        else:
            rewards[i].append(state.reward)
    if filter:
        for step, value_step in enumerate(rewards):
            rewards[step] = list(filter(lambda x: x != 0, value_step))
    mean_rewards = [np.mean(on_step_rewards) for on_step_rewards in rewards]
    arr_mean_rewards.append(mean_rewards[0:27])
    for report in reporter[1:3]:
        rewards = []
        for state in report.seen_states.state_list: 
            i = state.step
            if len(rewards) == i:
                rewards.append([state.reward])
            else:
                rewards[i].append(state.reward)
        if filter:
            for step, value_step in enumerate(rewards):
                rewards[step] = list(filter(lambda x: x != 0, value_step))
        mean_rewards = [np.mean(on_step_rewards) for on_step_rewards in rewards]
        arr_mean_rewards.append(mean_rewards)
    plt.figure()
    plt.xlabel("Steps")
    plt.ylabel("Rewards")
    plt.title(f"Non-terminal rules: {reporter[1].non_terminal_rules_limit}. Object: box")
    for m_reward in arr_mean_rewards:
        plt.plot(m_reward)
    plt.legend(legend, loc="upper left")
    plt.savefig(path, format="svg")

if __name__== "__main__":
    # report_1, __, __ = c.configure_graph_old_zateynik()
    # report_2, control_optimizer, __ = c.configure_graph_old_ellipsoid()
    # report_3, control_optimizer, __ = c.configure_graph_new_ellipsoid()
    # report_2, control_optimizer, __ = c.configure_graph_old_cylinder()
    # report_3, control_optimizer, __ = c.configure_graph_new_cylinder()
    report_2, control_optimizer, __ = c.configure_graph_old_brusok()
    report_3, control_optimizer, __ = c.configure_graph_new_brusok()
    # report_1, control_optimizer, __ = c.configure_old_puck()
    # report_2, control_optimizer, __ = c.configure_graph_old_puck()
    # report_3, control_optimizer, __ = c.configure_graph_new_puck()
    top_ten_best_render(report_2, control_optimizer, (11,15))
    # one_best_render(report_1, control_optimizer)
    # save_svg_mean_reward((report_1, report_2, report_3), ["AOC", "ACG", "SCG"], "01_03_23_graph_compare_rules_brusok_skum_1")
    
    # report_1, control_optimizer, __ = c.configure_graph_new_ellipsoid()
    # report_2, control_optimizer, __ = c.configure_graph_new_puck()
    # report_3, control_optimizer, __ = c.configure_graph_new_cylinder()
    # report_4, control_optimizer, __ = c.configure_graph_new_brusok()
    # report_1, control_optimizer, __ = c.configure_graph_old_ellipsoid()
    # report_2, control_optimizer, __ = c.configure_graph_old_puck()
    # report_3, control_optimizer, __ = c.configure_graph_old_cylinder()
    # report_4, control_optimizer, __ = c.configure_graph_old_brusok()
    # save_svg_mean_reward((report_1, report_2, report_3, report_4),
    #                      ["ellipsoid", "puck", "cylinder", "box"],
    #                      "01_03_23_graph_compare_objects_oldrule_MCTS")