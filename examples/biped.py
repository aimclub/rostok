from rostok.simulation_chrono.simulation_scenario import WalkingScenario
from rostok.library.rule_sets.leg_rules import get_biped

scenario = WalkingScenario(0.00001, 20)

graph = get_biped()

control = []

scenario.run_simulation(graph, control, starting_positions=[[0,0,0,0], [0,0,0,0]])
