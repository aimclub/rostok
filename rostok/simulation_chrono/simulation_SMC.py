import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import pychrono as chrono
import pychrono.irrlicht as chronoirr

from rostok.block_builder_api.block_parameters import (DefaultFrame,
                                                       FrameTransform)
from rostok.block_builder_chrono.block_classes import ChronoEasyShapeObject
from rostok.control_chrono.controller import (ConstController,
                                              ForceControllerTemplate,
                                              ForceTorqueContainer,
                                              YaxisShaker)
from rostok.criterion.simulation_flags import (EventCommands,
                                               SimulationSingleEvent)
from rostok.graph_grammar.node import GraphGrammar
from rostok.simulation_chrono.basic_simulation import SimulationResult
from rostok.virtual_experiment.robot_new import BuiltGraphChrono, RobotChrono
from rostok.virtual_experiment.sensors import DataStorage, Sensor


class ChronoSystems():

    @staticmethod
    def chrono_SMC_system(gravity_list=[0, 0, 0]):
        system = chrono.ChSystemSMC()
        system.UseMaterialProperties(False)
        system.SetSolverMaxIterations(1000)
        system.SetSolverForceTolerance(1e-4)
        system.Set_G_acc(chrono.ChVectorD(gravity_list[0], gravity_list[1], gravity_list[2]))
        # time_stepper = chrono.ChTimestepperHHT()
        # time_stepper.SetMaxiters(4)
        # time_stepper.SetMinStepSize(1e-4)
        # system.SetTimestepper(time_stepper)
        # stepper = system.GetTimestepper()
        #system.SetTimestepperType(chrono.ChTimestepper.Type_HHT)
        # stepper = system.GetTimestepper()
        # stepper = chrono.ChTimestepperHHT(stepper)
        system.SetContactForceModel(0)
        return system

    @staticmethod
    def chrono_NSC_system(gravity_list=[0, 0, 0]):
        system = chrono.ChSystemNSC()
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.SetSolverMaxIterations(100)
        system.SetSolverForceTolerance(1e-6)
        system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        system.Set_G_acc(chrono.ChVectorD(gravity_list[0], gravity_list[1], gravity_list[2]))
        return system


class EnvCreator():

    def __init__(self, object_list: List[Tuple[ChronoEasyShapeObject, bool]] = []):
        self.objects: List[ChronoEasyShapeObject] = []
        self.active_body_counter = 0
        self.active_objects_ordered: Dict[int, ChronoEasyShapeObject] = {}
        self.force_torque_container = ForceTorqueContainer()
        self.env_data_dict = {}
        for obj in object_list:
            self.add_object(obj[0], obj[1])

    def add_env_data_type_dict(self, data_dict):
        self.env_data_dict = data_dict

    def add_object(self,
                   obj: ChronoEasyShapeObject,
                   read_data: bool = False,
                   is_fixed=False,
                   force_torque_controller: Optional[ForceControllerTemplate] = None):
        """" Add an object to the environment
        
            Args:
                obj (ChronoEasyShapeObject): object description and chrono body
                read_data (bool): define if we add a body to env_sensor
                is_fixed (bool): define if the object is fixed"""
        if is_fixed:
            obj.body.SetBodyFixed(True)

        self.objects.append(obj)
        if force_torque_controller:
            force_torque_controller.bind_body(obj.body)
            self.force_torque_container.add(force_torque_controller)
        if read_data:
            self.active_objects_ordered[self.active_body_counter] = obj
            self.active_body_counter += 1

    def build_data_storage(self, max_number_of_steps) -> None:
        """Initialize Sensor for environment and data stores for robot and environment

            Args:
                max_number_of_steps (int): maximum number of steps in the simulation"""

        env_sensor: Sensor = Sensor(self.active_objects_ordered, {})
        env_sensor.contact_reporter.reset_contact_dict()
        self.data_storage: DataStorage = DataStorage(env_sensor)
        for key, value in self.env_data_dict.items():
            self.data_storage.add_data_type(key, value[0], value[1], max_number_of_steps)

    def load_into_system(self, system: chrono.ChSystem):
        for obj in self.objects:
            system.AddBody(obj.body)


class ChronoVisManager():

    def __init__(self, delay: bool = False):
        self.vis = chronoirr.ChVisualSystemIrrlicht()
        self.delay_flag = delay

    def initialize_vis(self, chrono_system):
        self.vis.AttachSystem(chrono_system)
        self.vis.SetWindowSize(1024, 768)
        self.vis.SetWindowTitle('Grab demo')
        self.vis.Initialize()
        self.vis.AddSkyBox()
        self.vis.AddCamera(chrono.ChVectorD(-0.15, 0.30, 0.40))
        self.vis.AddTypicalLights()
        #self.vis.EnableCollisionShapeDrawing(True)


class SingleRobotSimulation():

    def __init__(self, system: chrono.ChSystem, env_creator: EnvCreator,
                 vis_manager: ChronoVisManager):
        self.chrono_system = system
        self.env_creator = env_creator
        self.vis_manager = vis_manager
        self.result = SimulationResult()
        self.robot_data_dict = {}

    def add_robot_data_type_dict(self, data_dict):
        self.robot_data_dict = data_dict

    def initialize(self, max_number_of_steps: int):
        self.env_creator.build_data_storage(max_number_of_steps)
        self.env_creator.load_into_system(self.chrono_system)

        for key, value in self.robot_data_dict.items():
            self.robot.data_storage.add_data_type(key, value[0], value[1], max_number_of_steps)

    def add_design(self,
                   graph: GraphGrammar,
                   control_parameters,
                   control_cls=ConstController,
                   Frame: FrameTransform = DefaultFrame,
                   starting_positions=[],
                   is_fixed=True):
        """Add a robot to simulation using graph and control parameters

            Args:
                graph (GraphGrammar): graph of the robot
                control_parameters: parameters for the controller
                control_cls: controller class
                Frame (FrameTransform): initial coordinates of the base body of the robot
                is_fixed (bool): define if the base body is fixed
                with_data (bool): define if we store sensor data for robot
        """
        self.robot = RobotChrono(graph, self.chrono_system, control_parameters, control_cls, Frame,
                                 starting_positions, is_fixed)

    def update_data(self, step_n):
        """Update the env_sensor and env_data.
            Args:
                step_n (int): number of the current step"""
        self.env_creator.data_storage.sensor.contact_reporter.reset_contact_dict()
        self.env_creator.data_storage.sensor.update_current_contact_info(self.chrono_system)
        self.env_creator.data_storage.update_storage(step_n)

    def simulate_step(self, step_length: float, current_time: float, step_n: int):
        """Simulate one step and update sensors and data stores
        
            Args:
                step_length (float): the time of the step
                current_time (float): current time of the simulation
                step_n: number of the current step"""

        self.chrono_system.Update()
        self.chrono_system.DoStepDynamics(step_length)
        self.update_data(step_n)

        robot: RobotChrono = self.robot
        ds = robot.data_storage
        robot.sensor.contact_reporter.reset_contact_dict()
        robot.sensor.update_current_contact_info(self.chrono_system)
        robot.data_storage.update_storage(step_n)

        #controller gets current states of the robot and environment and updates control functions
        robot.controller.update_functions(current_time, robot.sensor,
                                          self.env_creator.data_storage.sensor)
        self.env_creator.force_torque_container.update_all(current_time,
                                                           self.env_creator.data_storage.sensor)

    def activate(self, current_time):
        self.env_creator.force_torque_container.controller_list[0].start_time = current_time

    def handle_single_events(self, event_container, current_time, step_n):
        if event_container is None:
            return False

        for event in event_container:
            if not event.state:
                event_command = event.event_check(current_time, step_n, self.robot.sensor,
                                                  self.env_creator.data_storage.sensor)
                if event_command == EventCommands.STOP:
                    return True
                elif event_command == EventCommands.ACTIVATE:
                    self.activate(current_time)

        return False

    def simulate(
        self,
        number_of_steps: int,
        step_length: float,
        fps: int = 100,
        event_container=None,
        visualize=False,
    ):
        """Execute a simulation.
        
            Args:
                number_of_steps(int): total number of steps in the simulation
                step_length (float): the time length of a step
                frame_update (int): rate of visualization update
                flag_container: container of flags that controls simulation
                visualize (bool): determine if run the visualization """
        self.initialize(number_of_steps)

        if visualize:
            self.vis_manager.initialize_vis(self.chrono_system)

        stop_flag = False
        self.result.time_vector = [0]
        frame_simulation = 0
        for i in range(number_of_steps):
            current_time = self.chrono_system.GetChTime()
            self.simulate_step(step_length, current_time, i)
            self.result.time_vector.append(self.chrono_system.GetChTime())
            if visualize:
                if frame_simulation > 1 / fps / step_length:
                    frame_simulation = 0
                    self.vis_manager.vis.Run()
                    self.vis_manager.vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
                    self.vis_manager.vis.Render()
                    self.vis_manager.vis.EndScene()
                    # just to slow down the simulation
                    if self.vis_manager.delay_flag:
                        time.sleep(0.000001)
                else:
                    frame_simulation += 1
            else:
                if frame_simulation > 1 / fps / step_length:
                    frame_simulation = 0
                    # print(i)
                else:
                    frame_simulation += 1
            stop_flag = self.handle_single_events(event_container, current_time, i)
            if stop_flag:
                break

        if visualize:
            self.vis_manager.vis.GetDevice().closeDevice()
        self.result.environment_final_ds = self.env_creator.data_storage
        self.result.robot_final_ds = self.robot.data_storage
        self.result.time = self.chrono_system.GetChTime()
        self.n_steps = number_of_steps
        self.result.reduce_ending(i)
        return self.result
