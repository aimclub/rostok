from rostok.simulation_chrono.simulation_scenario import WalkingScenario
from rostok.library.rule_sets.leg_rules import get_biped
from rostok.library.rule_sets.ruleset_simple_wheels import get_four_wheels
from rostok.graph_grammar.graph_utils import plot_graph

scenario = WalkingScenario(0.0001, 3)

graph = get_biped()
graph = get_four_wheels()
plot_graph(graph)
control = control_parameters = {"initial_value": [0.02]*8}

scenario.run_simulation(graph, control, starting_positions=None, vis = True, delay=True)
