from mcts_run_setup import config_with_standard_graph
import matplotlib.pyplot as plt
from pathlib import Path
from rostok.graph_generators.mcts_helper import OptimizedGraphReport
from rostok.utils.pickle_save import load_saveable
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
import networkx as nx
def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()


rule_vocabul, torque_dict = create_rules()

grasp_object_blueprint = get_object_parametrized_sphere(0.4, 0.7)

report = load_saveable(Path("results\Reports_23y_02m_22d_04H_51M\MCTS_data_windows.pickle"))

control_optimizer = config_with_standard_graph(grasp_object_blueprint, torque_dict)

best_graph, reward, best_control = report.get_best_info()

simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_control

plot_graph(best_graph)
data = {"initial_value": best_control}
simulation_output = simulation_manager.run_simulation(best_graph, data, True)
res = -simulation_rewarder.calculate_reward(simulation_output)
print(res)


main_graph, reward, main_control = report.get_main_info()

plot_graph(main_graph)
data = {"initial_value": main_control}
simulation_output = simulation_manager.run_simulation(main_graph, data, True)
res = -simulation_rewarder.calculate_reward(simulation_output)
print(res)
