from rostok.simulation_chrono.simulation_scenario import WalkingScenario

from wheels import get_wheels
from rostok.graph_grammar.graph_utils import plot_graph

scenario = WalkingScenario(0.0001, 3)
graph = get_wheels()

control = control_parameters = {"initial_value": [0.05]*2}

scenario.run_simulation(graph, control, starting_positions=[[45,-90,0], [-90,90,0], [90,-90,0]], vis = True, delay=True)

