import types
import re
from typing import Dict, List, Optional, Tuple

import pychrono as chrono
import numpy as np

from rostok.criterion.simulation_flags import SimulationSingleEvent
from rostok.graph_grammar.node import GraphGrammar
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono, RobotSimulationWithForceTest
from rostok.virtual_experiment.sensors import (SensorCalls, SensorObjectClassification)
from rostok.simulation_chrono.simulation_utils import set_covering_sphere_based_position
from rostok.control_chrono.controller import ConstController, SinControllerChrono, YaxisShaker


class ParametrizedSimulation:

    def __init__(self, step_length, simulation_length):
        self.step_length = step_length
        self.simulation_length = simulation_length

    def run_simulation(self, graph: GraphGrammar, data):
        pass
    
    def __repr__(self) -> str:
        str_type = str(type(self))
        str_class = re.findall('\'([^\']*)\'', str_type)[0]
        self_attributes = dir(self)
        self_fields = list(filter(lambda x: not (x.startswith("__") or x.endswith("__")), self_attributes))
        self_fields = list(filter(lambda x: not isinstance(getattr(self, x), types.MethodType), self_fields))
        str_self = f"{str_class}:\n"
        for str_field in self_fields:
            str_self = str_self + f"    {str_field} = {getattr(self, str_field)}, \n"
        return str_self


class ConstTorqueGrasp(ParametrizedSimulation):

    def __init__(self, step_length, simulation_length) -> None:
        super().__init__(step_length, simulation_length)
        self.grasp_object_callback = None
        self.event_container: List[SimulationSingleEvent] = []

    def add_event(self, event):
        self.event_container.append(event)

    def reset_events(self):
        for event in self.event_container:
            event.reset()

    def run_simulation(self, graph: GraphGrammar, data, vis=False):
        self.reset_events()
        #simulation = RobotSimulationChrono([])
        simulation = RobotSimulationWithForceTest(False, [])
        simulation.add_design(graph, data)
        grasp_object = self.grasp_object_callback()
        shake = YaxisShaker(10, 1, 0.5, float("inf"))
        set_covering_sphere_based_position(grasp_object,
                                           reference_point=chrono.ChVectorD(0, 0.05, 0))
        simulation.add_object(grasp_object, read_data=True, force_torque_controller=shake)
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
        return simulation.simulate(n_steps, self.step_length, 10, self.event_container, vis)