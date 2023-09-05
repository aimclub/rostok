import json
from typing import Dict, List, Optional, Tuple

import pychrono as chrono
import numpy as np

from rostok.criterion.simulation_flags import SimulationSingleEvent
from rostok.graph_grammar.node import GraphGrammar
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono, RobotSimulationWithForceTest
from rostok.virtual_experiment.sensors import (SensorCalls, SensorObjectClassification)
from rostok.simulation_chrono.simulation_utils import set_covering_sphere_based_position
from rostok.control_chrono.controller import ConstController, SinControllerChrono, YaxisShaker
from rostok.utils.json_encoder import RostokJSONEncoder
from rostok.simulation_chrono.simulation_SMC import SingleRobotSimulation, ChronoVisManager, EnvCreator, ChronoSystems


class ParametrizedSimulation:

    def __init__(self, step_length, simulation_length):
        self.step_length = step_length
        self.simulation_length = simulation_length

    def run_simulation(self, graph: GraphGrammar, data):
        pass

    def __repr__(self) -> str:
        json_data = json.dumps(self, cls=RostokJSONEncoder)
        return json_data

    def __str__(self) -> str:
        json_data = json.dumps(self, indent=4, cls=RostokJSONEncoder)
        return json_data


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

    def run_simulation(self, graph: GraphGrammar, data, vis=False, delay=False):
        self.reset_events()
        #simulation = RobotSimulationChrono([])
        simulation = RobotSimulationWithForceTest(delay, [])
        simulation.add_design(graph, data)
        grasp_object = self.grasp_object_callback()
        shake = YaxisShaker(1, 1, 0.5, float("inf"))
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

from rostok.control_chrono.tendon_controller import TendonController_2p
class SMCGrasp(ParametrizedSimulation):

    def __init__(self, step_length, simulation_length, tendon = True) -> None:
        super().__init__(step_length, simulation_length)
        self.grasp_object_callback = None
        self.event_container: List[SimulationSingleEvent] = []
        self.tendon = tendon

    def add_event(self, event):
        self.event_container.append(event)

    def reset_events(self):
        for event in self.event_container:
            event.reset()

    def run_simulation(self, graph: GraphGrammar, data, starting_positions = [],vis=False, delay=False):
        self.reset_events()
        # build simulation from the subclasses
        #system = ChronoSystems.chrono_SMC_system([0, -10, 0])
        system = ChronoSystems.chrono_NSC_system([0, -10, 0])
        env_creator = EnvCreator([])
        vis_manager = ChronoVisManager(delay)
        simulation = SingleRobotSimulation(system, env_creator, vis_manager)
        
        
        

        grasp_object = self.grasp_object_callback()
        shake = YaxisShaker(1, 3, 0.5, float("inf"))
        # the object  positioning based on the AABB
        set_covering_sphere_based_position(grasp_object,
                                           reference_point=chrono.ChVectorD(0, 0.05, 0))
        simulation.env_creator.add_object(grasp_object,
                                          read_data=True,
                                          force_torque_controller=shake)
        # add design and determine the outer force
        if self.tendon:
            simulation.add_design(graph, data, TendonController_2p, starting_positions=starting_positions)
        else:
            simulation.add_design(graph, data, starting_positions=starting_positions)
        # setup parameters for the data store
        
        n_steps = int(self.simulation_length / self.step_length)
        env_data_dict = {
            "n_contacts": (SensorCalls.AMOUNT_FORCE, SensorObjectClassification.BODY),
            "forces": (SensorCalls.FORCE, SensorObjectClassification.BODY),
            "COG": (SensorCalls.BODY_TRAJECTORY, SensorObjectClassification.BODY,
                    SensorCalls.BODY_TRAJECTORY),
            "force_center": (SensorCalls.FORCE_CENTER, SensorObjectClassification.BODY)
        }
        simulation.env_creator.add_env_data_type_dict(env_data_dict)
        robot_data_dict = {
            "body_velocity": (SensorCalls.BODY_VELOCITY, SensorObjectClassification.BODY,
                              SensorCalls.BODY_VELOCITY),
            "COG": (SensorCalls.BODY_TRAJECTORY, SensorObjectClassification.BODY,
                    SensorCalls.BODY_TRAJECTORY),
            "n_contacts": (SensorCalls.AMOUNT_FORCE, SensorObjectClassification.BODY)
        }
        simulation.add_robot_data_type_dict(robot_data_dict)
        return simulation.simulate(n_steps, self.step_length, 10000, self.event_container, vis)
