 
from rostok.graph_generators.mcts_helper import MCTSSaveable
from mcts_run_setup import config_with_standard_graph
import matplotlib.pyplot as plt
from pathlib import Path
from rostok.graph_generators.mcts_helper import OptimizedGraphReport
from rostok.utils.pickle_save import load_saveable
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_easy_box
import networkx as nx
from statistics import mean
import matplotlib.pyplot as plt

def plot_means(report):
    """Plot the mean rewards for steps of MCTS search."""

    rewards = []
    for state in report.seen_states.state_list:
        i = state.step
        if len(rewards) == i:
            rewards.append([state.reward])
        else:
            rewards[i].append(state.reward)

        mean_rewards = [mean(sorted(on_step_rewards)[-5:]) for on_step_rewards in rewards]
        #mean_rewards = [mean(on_step_rewards) for on_step_rewards in rewards]
    
    plt.plot(mean_rewards)

def plot_list_step(report_list):
  
    for i in report_list:
        plot_means(i)
    

path_1 = "results\Reports_23y_06m_08d_18H_43M\MCTS_data.pickle"
path_2 = "results\Reports_23y_06m_08d_18H_11M\MCTS_data.pickle"
path_3 = "results\Reports_23y_06m_08d_17H_56M\MCTS_data.pickle"

report1 = load_saveable(Path(path_1))
report2 = load_saveable(Path(path_2))
report3 = load_saveable(Path(path_3))

reports_label = ["cylinder", "box", "ellipsoid"]

reports = [report1, report2, report3]
plt.figure()
plot_list_step(reports)
plt.legend(reports_label)
plt.xlabel('Steps')
plt.ylabel('Mean rewards 5')
plt.show()


 