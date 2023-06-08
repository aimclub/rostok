from mcts_run_setup import config_with_standard_graph
import matplotlib.pyplot as plt
from pathlib import Path
from rostok.graph_generators.mcts_helper import OptimizedGraphReport
from rostok.utils.pickle_save import load_saveable
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.library.obj_grasp.objects import get_object_parametrized_cylinder, get_object_easy_box, get_obj_hard_get_obj_hard_large_ellipsoid
import networkx as nx

obj_ell = get_obj_hard_get_obj_hard_large_ellipsoid()
grasp_object_cyl = get_object_parametrized_cylinder(0.4, 1, 0.7)
rule_vocabul, torque_dict = create_rules()

grasp_object_blueprint = obj_ell

report = load_saveable(Path("results\Reports_23y_06m_08d_17H_56M\MCTS_data.pickle"))

control_optimizer = config_with_standard_graph(grasp_object_blueprint, torque_dict)

simulation_rewarder = control_optimizer.rewarder
simulation_manager = control_optimizer.simulation_control

def sorter(x):
    return -x.reward
graph_list_top = sorted(report.seen_graphs.graph_list, key=sorter)[0:5]

for big_g in graph_list_top:
    control_value = big_g.control
    data = {"initial_value": control_value}
    simulation_output = simulation_manager.run_simulation(big_g.graph, data, False)
    res = simulation_rewarder.calculate_reward(simulation_output)
    print(res)


