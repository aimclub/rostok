
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono, SystemPreviewChrono
import random
from rostok.graph_grammar.node import GraphGrammar

def mock_with_build_mech(graph: GraphGrammar):
    # Build graph
    sim = SystemPreviewChrono()
    sim.add_design(graph)
    sim.simulate_step(0.001)
    return random.random()*5

