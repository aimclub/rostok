import types
from typing import Dict, List, Optional, Tuple

import pychrono as chrono
import numpy as np

from rostok.criterion.simulation_flags import FlagStopSimualtions
from rostok.graph_grammar.node import GraphGrammar
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono
from rostok.virtual_experiment.sensors import (SensorCalls, SensorObjectClassification)
from rostok.simulation_chrono.simulation_utils import set_covering_sphere_based_position


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

    def run_simulation(self, graph: GraphGrammar, data, vis=True):
        self.reset_flags()
        simulation = RobotSimulationChrono([])
        simulation.add_design(graph, data)
        grasp_object = self.grasp_object_callback()
        set_covering_sphere_based_position(grasp_object,
                                           reference_point=chrono.ChVectorD(0, 0.05, 0))
        simulation.add_object(grasp_object, read_data=True)
        n_steps = int(self.simulation_length / self.step_length)
        env_data_dict = {
            "n_contacts": (SensorCalls.AMOUNT_FORCE, SensorObjectClassification.BODY),
            "forces": (SensorCalls.FORCE, SensorObjectClassification.BODY),
            "COG": (SensorCalls.BODY_TRAJECTORY, SensorObjectClassification.BODY,
                    SensorCalls.BODY_TRAJECTORY),
            "force_center": (SensorCalls.FORCE_CENTER, SensorObjectClassification.BODY)
        }
        simulation.add_env_data_type_dict(env_data_dict)
        robot_data_dict = {
            "n_contacts": (SensorCalls.AMOUNT_FORCE, SensorObjectClassification.BODY)
        }
        simulation.add_robot_data_type_dict(robot_data_dict)
        return simulation.simulate(n_steps, self.step_length, 10, self.flag_container, vis)