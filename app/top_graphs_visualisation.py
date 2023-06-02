from pathlib import Path

from mcts_run_setup import config_with_standard_graph, config_with_standard

from rostok.graph_generators.mcts_helper import OptimizedGraphReport
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
#from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.utils.pickle_save import load_saveable

rule_vocabul = create_rules()

grasp_object_blueprint = get_object_parametrized_sphere(0.2, 1)

graph_report: OptimizedGraphReport = load_saveable(Path(r"results\Reports_23y_05m_31d_19H_17M\optimized_graph_report.pickle"))

control_optimizer = config_with_standard(grasp_object_blueprint)
simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_control
graph_list = graph_report.graph_list

sorted_graph_list = sorted(graph_list, key = lambda x: x.reward)
some_top = sorted_graph_list[-1:-2:-1]
for graph in some_top:
    G = graph.graph
    reward = graph.reward
    control = graph.control

    _, _ = control_optimizer.count_reward(G)
    #control = control.round(3)
    data = {"initial_value": control}
    simulation_output = simulation_manager.run_simulation(G, data, True)
    res = -simulation_rewarder.calculate_reward(simulation_output)
    print(reward)
    print(res)
    print()
