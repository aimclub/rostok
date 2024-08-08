from copy import deepcopy
import json
from typing import List, Optional

import pychrono as chrono

from rostok.control_chrono.controller import (ConstController, SinControllerChrono)
from rostok.control_chrono.external_force import ForceChronoWrapper, ABCForceCalculator
from rostok.criterion.simulation_flags import EventBuilder
from rostok.graph_grammar.node import GraphGrammar
from rostok.simulation_chrono.simulation import (ChronoSystems, EnvCreator, SingleRobotSimulation,
                                                 ChronoVisManager)
from rostok.simulation_chrono.simulation_utils import \
    set_covering_ellipsoid_based_position
from rostok.utils.json_encoder import RostokJSONEncoder
from rostok.virtual_experiment.sensors import (SensorCalls, SensorObjectClassification)
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.control_chrono.tendon_controller import TendonController_2p
from abc import abstractmethod

class ParametrizedSimulation:

    def __init__(self, step_length, simulation_length):
        self.step_length = step_length
        self.simulation_length = simulation_length

    def run_simulation(self,
                       graph: GraphGrammar,
                       controller_data,
                       starting_positions=None,
                       vis=False,
                       delay=False):
        pass

    def __repr__(self) -> str:
        json_data = json.dumps(self, cls=RostokJSONEncoder)
        return json_data

    def __str__(self) -> str:
        json_data = json.dumps(self, indent=4, cls=RostokJSONEncoder)
        return json_data
    
    @abstractmethod
    def get_scenario_name(self):
        return self.__str__


class GraspScenario(ParametrizedSimulation):
    def __init__(self,
                 step_length,
                 simulation_length,
                 controller_cls = ConstController,
                 smc=False,
                 obj_external_forces: Optional[ABCForceCalculator] = None) -> None:
        super().__init__(step_length, simulation_length)
        self.grasp_object_callback = None
        self.event_builder_container: List[EventBuilder] = []
        self.controller_cls = controller_cls
        self.smc = smc
        self.obj_external_forces = obj_external_forces

    def add_event_builder(self, event_builder):
        self.event_builder_container.append(event_builder)

    def build_events(self):
        event_list=[]
        for event_builder in self.event_builder_container:
            event_builder.build_event(event_list)
        return event_list

    def run_simulation(self,
                       graph: GraphGrammar,
                       controller_data,
                       starting_positions=None,
                       vis=False,
                       delay=False):
        # events should be reset before every simulation
        event_list = self.build_events()
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
        if self.obj_external_forces:
            chrono_forces = ForceChronoWrapper(deepcopy(self.obj_external_forces), event_list)
        else:
            chrono_forces = None
        simulation.env_creator.add_object(grasp_object,
                                          read_data=True,
                                          force_torque_controller=chrono_forces)

        # add design and determine the outer force

        simulation.add_design(graph,
                                controller_data,
                                self.controller_cls,
                                starting_positions=starting_positions)
         
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
        return simulation.simulate(n_steps, self.step_length, 10000, event_list, vis)
    
    def get_scenario_name(self):
        return str(self.grasp_object_callback)


from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.utils.dataset_materials.material_dataclass_manipulating import (
    DefaultChronoMaterialNSC, DefaultChronoMaterialSMC)
from rostok.block_builder_api.block_parameters import (DefaultFrame, FrameTransform)
class WalkingScenario(ParametrizedSimulation):
    def __init__(self,
                 step_length,
                 simulation_length,
                 controller_cls = ConstController,
                 smc=False,) -> None:
        super().__init__(step_length, simulation_length)

        self.event_builder_container: List[EventBuilder] = []
        self.controller_cls = controller_cls
        self.smc = smc

    def add_event_builder(self, event_builder):
        self.event_builder_container.append(event_builder)

    def build_events(self):
        event_list=[]
        for event_builder in self.event_builder_container:
            event_builder.build_event(event_list)
        return event_list

    def run_simulation(self,
                       graph: GraphGrammar,
                       controller_data,
                       starting_positions=None,
                       vis=False,
                       delay=False):
        # events should be reset before every simulation
        event_list = self.build_events()
        # build simulation from the subclasses

        if self.smc:
            system = ChronoSystems.chrono_SMC_system(gravity_list=[0, 0, 0])
        else:
            system = ChronoSystems.chrono_NSC_system(gravity_list=[0, -1, 0])
        # setup the auxiliary
        env_creator = EnvCreator([])
        vis_manager = ChronoVisManager(delay)
        simulation = SingleRobotSimulation(system, env_creator, vis_manager)

        if self.smc:
            def_mat = DefaultChronoMaterialSMC()
        else:
            def_mat = DefaultChronoMaterialNSC()
        floor = creator.create_environment_body(EnvironmentBodyBlueprint(Box(1, 0.1, 1), material=def_mat, color=[215, 255, 0]))
        floor.body.SetNameString("Floor")
        floor.body.SetPos(chrono.ChVectorD(0,-0.05,0))
        #floor.body.GetVisualShape(0).SetTexture("/home/yefim-work/Packages/miniconda3/envs/rostok/share/chrono/data/textures/bluewhite.png", 10, 10)
        floor.body.SetBodyFixed(True)


        simulation.env_creator.add_object(floor,
                                          read_data=True,
                                          is_fixed=True)

        # add design and determine the outer force

        # simulation.add_design(graph,
        #                         controller_data,
        #                         self.controller_cls,
        #                         Frame=FrameTransform([0, 0.25, 0], [3**0.5/2, 0, 0, 1/2]),
        #                         starting_positions=starting_positions, is_fixed=False)
        simulation.add_design(graph,
                                controller_data,
                                self.controller_cls,
                                Frame=FrameTransform([0, 0.25, 0], [0,0,0,1]),
                                starting_positions=starting_positions, is_fixed=False)
         
        # setup parameters for the data store

        n_steps = int(self.simulation_length / self.step_length)
        env_data_dict = {

        }
        simulation.env_creator.add_env_data_type_dict(env_data_dict)
        robot_data_dict = {
        }
        simulation.add_robot_data_type_dict(robot_data_dict)
        return simulation.simulate(n_steps, self.step_length, 10000, event_list, vis)
    
    def get_scenario_name(self):
        return "Moving robot"



class SuspensionCarScenario(ParametrizedSimulation):
    def __init__(self,
                 step_length,
                 simulation_length,
                 controller_cls = ConstController,
                 smc=False) -> None:
        super().__init__(step_length, simulation_length)
        self.event_builder_container: List[EventBuilder] = []
        self.controller_cls = controller_cls
        self.smc = smc

    def add_event_builder(self, event_builder):
        self.event_builder_container.append(event_builder)

    def build_events(self):
        event_list=[]
        for event_builder in self.event_builder_container:
            event_builder.build_event(event_list)
        return event_list

    def run_simulation(self,
                       graph: GraphGrammar,
                       controller_data,
                       starting_positions=None,
                       vis=False,
                       delay=False):
        # events should be reset before every simulation
        event_list = self.build_events()
        # build simulation from the subclasses

        if self.smc:
            system = ChronoSystems.chrono_SMC_system(gravity_list=[0, 0, 0])
        else:
            system = ChronoSystems.chrono_NSC_system(gravity_list=[0, -10, 0])
        # setup the auxiliary
        env_creator = EnvCreator([])
        vis_manager = ChronoVisManager(delay)
        simulation = SingleRobotSimulation(system, env_creator, vis_manager)

        if self.smc:
            def_mat = DefaultChronoMaterialSMC()
        else:
            def_mat = DefaultChronoMaterialNSC()
        floor = creator.create_environment_body(EnvironmentBodyBlueprint(Box(1, 0.1, 1), material=def_mat, color=[215, 255, 0]))
        floor.body.SetNameString("Floor")
        floor.body.SetPos(chrono.ChVectorD(0,-0.05,0))
        #floor.body.GetVisualShape(0).SetTexture("/home/yefim-work/Packages/miniconda3/envs/rostok/share/chrono/data/textures/bluewhite.png", 10, 10)
        floor.body.SetBodyFixed(True)


        simulation.env_creator.add_object(floor,
                                          read_data=True,
                                          is_fixed=True)

        simulation.add_design(graph,
                                controller_data,
                                self.controller_cls,
                                Frame=FrameTransform([0, 0.25, 0], [0,0,0,1]),
                                starting_positions=starting_positions, is_fixed=False)
         
        # setup parameters for the data store

        n_steps = int(self.simulation_length / self.step_length)
        env_data_dict = {

        }
        simulation.env_creator.add_env_data_type_dict(env_data_dict)
        robot_data_dict = {
            "COG": (SensorCalls.BODY_TRAJECTORY, SensorObjectClassification.BODY,
                    SensorCalls.BODY_TRAJECTORY),
        }
        simulation.add_robot_data_type_dict(robot_data_dict)
        return simulation.simulate(n_steps, self.step_length, 10000, event_list, vis)
    
    def get_scenario_name(self):
        return "Moving robot"