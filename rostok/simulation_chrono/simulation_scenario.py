from copy import deepcopy
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pychrono as chrono

from rostok.control_chrono.controller import (ConstController,
                                              SinControllerChrono)
from rostok.criterion.simulation_flags import SimulationSingleEvent
from rostok.graph_grammar.node import GraphGrammar
from rostok.simulation_chrono.simulation import (ChronoSystems, EnvCreator, SingleRobotSimulation,
                                                 ChronoVisManager)
from rostok.simulation_chrono.simulation_utils import \
    set_covering_sphere_based_position, set_covering_ellipsoid_based_position
from rostok.utils.json_encoder import RostokJSONEncoder
from rostok.virtual_experiment.sensors import (SensorCalls, SensorObjectClassification)
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.control_chrono.tendon_controller import TendonController_2p

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


class GraspScenario(ParametrizedSimulation):

    def __init__(self,
                 step_length,
                 simulation_length,
                 tendon=True,
                 smc=False,
                 obj_external_forces: Optional[ForceControllerTemplate] = None) -> None:
        super().__init__(step_length, simulation_length)
        self.grasp_object_callback = None
        self.event_container: List[SimulationSingleEvent] = []
        self.tendon = tendon
        self.smc = smc
        self.obj_external_forces = obj_external_forces

    def add_event(self, event):
        self.event_container.append(event)

    def reset_events(self):
        for event in self.event_container:
            event.reset()

    def run_simulation(self,
                       graph: GraphGrammar,
                       data,
                       starting_positions=None,
                       vis=False,
                       delay=False):
        # events should be reset before every simulation
        self.reset_events()
        # build simulation from the subclasses

        if self.smc:
            system = ChronoSystems.chrono_SMC_system(gravity_list=[0, 0, 0])
        else:
            system = ChronoSystems.chrono_NSC_system(gravity_list=[0, -10, 0])
        # setup the auxiliary
        env_creator = EnvCreator([])
        vis_manager = ChronoVisManager(delay)
        simulation = SingleRobotSimulation(system, env_creator, vis_manager)

        grasp_object = creator.create_environment_body(self.grasp_object_callback)
        grasp_object.body.SetNameString("Grasp_object")
        set_covering_ellipsoid_based_position(grasp_object,
                                           reference_point=chrono.ChVectorD(0, 0.1, 0))

        simulation.env_creator.add_object(grasp_object,
                                          read_data=True,
                                          force_torque_controller=None)

        # add design and determine the outer force
        if self.tendon:
            simulation.add_design(graph,
                                  data,
                                  TendonController_2p,
                                  starting_positions=starting_positions)
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