from mcts_run_setup import config_with_standard_graph, config_with_standard
import matplotlib.pyplot as plt
from pathlib import Path
from rostok.graph_generators.mcts_helper import OptimizedGraphReport
from rostok.graph_grammar.node import GraphGrammar
from rostok.utils.pickle_save import load_saveable
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_parametrized_tilt_ellipsoid, get_object_parametrized_cylinder
import networkx as nx
def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()


rule_vocabul, torque_dict = create_rules()

grasp_object_blueprint = get_object_parametrized_cylinder(0.5, 0.4, 0.2)

report = load_saveable(Path("results\Reports_23y_07m_20d_09H_05M\MCTS_data.pickle"))

control_optimizer = config_with_standard(grasp_object_blueprint)

best_graph, reward, best_control = report.get_best_info()

simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_scenario

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
