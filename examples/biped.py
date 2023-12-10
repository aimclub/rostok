from rostok.simulation_chrono.simulation_scenario import WalkingScenario
from rostok.library.rule_sets.leg_rules import get_biped
from rostok.graph_grammar.graph_utils import plot_graph

scenario = WalkingScenario(0.00001, 20)

graph = get_biped()
plot_graph(graph)
control = controll_parameters = {"initial_value": [0]*8}

scenario.run_simulation(graph, control, starting_positions=[[0,-30,60,-30], [0,30,-60,30]], vis = True, delay=True)
