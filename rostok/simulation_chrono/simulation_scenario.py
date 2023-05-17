from typing import Dict, List, Optional, Tuple

from rostok.criterion.simulation_flags import FlagStopSimualtions
from rostok.graph_grammar.node import GraphGrammar
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono


class ParametrizedSimulation:

    def __init__(self, step_length, simulation_length):
        self.step_length = step_length
        self.simulation_length = simulation_length

    def run_simulation(self, graph: GraphGrammar, data):
        pass


class ConstTorqueGrasp(ParametrizedSimulation):

    def __init__(self, step_length, simulation_length) -> None:
        super().__init__(step_length, simulation_length)
        self.grasp_object_callback = None
        self.flag_container: List[FlagStopSimualtions] = []

    def add_flag(self, flag):
        self.flag_container.append(flag)

    def reset_flags(self):
        for flag in self.flag_container:
            flag.reset_flag()

    def run_simulation(self, graph: GraphGrammar, data, vis = False):
        self.reset_flags()
        simulation = RobotSimulationChrono([])
        simulation.add_design(graph, data)
        grasp_object = self.grasp_object_callback()
        simulation.add_object(grasp_object, True)
        n_steps = int(self.simulation_length / self.step_length)
        return simulation.simulate(n_steps, self.step_length, 10, self.flag_container, vis)