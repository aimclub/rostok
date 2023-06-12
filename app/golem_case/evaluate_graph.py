from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono, SystemPreviewChrono
import random
from rostok.graph_grammar.node import GraphGrammar


def mock_with_build_mech(graph: GraphGrammar):
    """Build graph and do 2 simulation step

    Args:
        graph (GraphGrammar):

    Returns:
        Penalty: Random positive value [0:5]
    """
    # Build graph
    sim = SystemPreviewChrono()
    sim.add_design(graph)
    sim.simulate(2, False)
    return random.random() * 5
