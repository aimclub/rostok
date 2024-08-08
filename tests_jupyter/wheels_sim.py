from rostok.control_chrono.controller import SimpleKeyBoardController
from rostok.simulation_chrono.simulation_scenario import WalkingScenario

from wheels import get_stiff_wheels, get_wheels, get_stiff_wheels_ell
from rostok.graph_grammar.graph_utils import plot_graph


scenario = WalkingScenario(0.001, 10000, SimpleKeyBoardController)
graph = get_stiff_wheels_ell()

parameters = {}
parameters["forward"] = 0.5
parameters["reverse"]= 0.5
parameters["forward_rotate"] = 0.3
parameters["reverse_rotate"] = 0.2

 

scenario.run_simulation(graph, parameters, starting_positions=[[45,-90,0], [-90,90,0], [90,-90,0]], vis = True, delay=True, is_follow_camera = False)

