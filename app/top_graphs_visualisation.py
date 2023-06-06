from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mcts_run_setup import config_with_standard_graph, config_with_standard

from rostok.graph_generators.mcts_helper import OptimizedGraphReport, MCTSSaveable
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
#from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.utils.pickle_save import load_saveable
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint


def vis_top_n_mechs(report: MCTSSaveable, n: int, object:EnvironmentBodyBlueprint):
    graph_report = report.seen_graphs
    control_optimizer = config_with_standard(grasp_object_blueprint)
    simulation_rewarder = control_optimizer.rewarder
    simulation_manager = control_optimizer.simulation_control
    graph_list = graph_report.graph_list

    sorted_graph_list = sorted(graph_list, key = lambda x: x.reward)
    some_top = sorted_graph_list[-1:-n:-1]
    for graph in some_top:
        G = graph.graph
        reward = graph.reward
        control = graph.control
        _, _ = control_optimizer.count_reward(G)
        data = {"initial_value": control}
        simulation_output = simulation_manager.run_simulation(G, data, True)
        res = simulation_rewarder.calculate_reward(simulation_output)
        print(reward)
        print(res)
        print()

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
    plt.title(f"Non-terminal rules: {reporter[0].non_terminal_rules_limit}. Object: box")
    for m_reward in arr_mean_rewards:
        plt.plot(m_reward)
    plt.legend(legend, loc="upper left")
    plt.savefig(path, format="svg")

if __name__ == "__main__":
    rule_vocabul = create_rules()
    grasp_object_blueprint = get_object_parametrized_sphere(0.2, 1)
    report: OptimizedGraphReport = load_saveable(Path(r"results\Reports_23y_06m_02d_13H_14M\MCTS_data.pickle"))
    vis_top_n_mechs(report, 2, grasp_object_blueprint)
    save_svg_mean_reward([report], ["body"], 'body')