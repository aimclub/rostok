from pathlib import Path

from mcts_run_setup import config_with_standard_graph

from rostok.graph_generators.mcts_helper import OptimizedGraphReport
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.utils.pickle_save import load_saveable

rule_vocabul, torque_dict = create_rules()

grasp_object_blueprint = get_object_parametrized_sphere(0.4, 0.7)

graph_report: OptimizedGraphReport = load_saveable(Path(r"results\Reports_23y_05m_26d_15H_53M\optimized_graph_report.pickle"))

control_optimizer = config_with_standard_graph(grasp_object_blueprint, torque_dict)
simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_control
graph_list = graph_report.graph_list
reward_list = []
i_list = set()

top_list =[]
sorted_graph_list = sorted(graph_list, key = lambda x: x.reward)
some_top = sorted_graph_list[-1:-6:-1]
for graph in some_top:
    G = graph.graph
    reward = graph.reward
    control = graph.control
    data = {"initial_value": control}
    simulation_output = simulation_manager.run_simulation(G, data, True)
    res = -simulation_rewarder.calculate_reward(simulation_output)
    print(reward)
    print(res)
    print()
