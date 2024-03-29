from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pychrono as chrono
import pychrono.irrlicht as chronoirr

from rostok.block_builder_api.block_parameters import (DefaultFrame, FrameTransform)
from rostok.block_builder_chrono.block_classes import ChronoEasyShapeObject
from rostok.control_chrono.controller import ConstController, ForceControllerTemplate, ForceTorqueContainer, YaxisShaker
from rostok.graph_grammar.node import GraphGrammar
from rostok.virtual_experiment.robot_new import BuiltGraphChrono, RobotChrono
from rostok.virtual_experiment.sensors import DataStorage, Sensor
from rostok.criterion.simulation_flags import SimulationSingleEvent, EventCommands
import time


class SystemPreviewChrono:
    """A simulation of the motionless environment and design"""

    def __init__(self):
        """Initialize the chroon system with default parameters"""
        self.chrono_system = chrono.ChSystemNSC()
        self.chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.chrono_system.SetSolverMaxIterations(100)
        self.chrono_system.SetSolverForceTolerance(1e-6)
        self.chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0, -10, 0))

    def add_design(self, graph, frame: FrameTransform = DefaultFrame, is_fix_base=True):
        """Add a design into the system

            Args:
                graph (GraphGrammar): graph of the design
                frame (FrameTransform): initial position of the base body
        """

        BuiltGraphChrono(graph, self.chrono_system, frame, is_fix_base)

    def add_object(self, obj: ChronoEasyShapeObject):
        """Add an object to the environment.

            Args:
                obj (ChronoEasyShapeObject): one of the simple chrono objects"""

        self.chrono_system.AddBody(obj.body)

    def simulate_step(self, time_step: float):
        """Simulate one step"""
        self.chrono_system.Update()
        #self.chrono_system.DoStepDynamics(time_step)
        # TODO: add some check for collisions that can reveal the errors in objects or design positions.

    def simulate(self, number_of_steps: int, visualize: bool = True):
        """Simulate several steps and visualize system.

            The simulation purpose is to check the initial positions of objects and visualize the 
                environment and the mech design. More steps == longer simulation.

            Args:
                number_of_steps (int): the number of steps for simulation
                visualize (bool): the flag for visualization
        """
        #TODO: try to replace the number_of_steps for the stop by a button.
        vis = None
        if visualize:
            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(self.chrono_system)
            vis.SetWindowSize(1024, 768)
            vis.SetWindowTitle('Grab demo')
            vis.Initialize()

            vis.AddCamera(chrono.ChVectorD(1.5, 3, -2))
            vis.AddTypicalLights()
            #vis.EnableCollisionShapeDrawing(True)
        self.chrono_system.Update()
        self.chrono_system.DoStepDynamics(1e-4)
        for _ in range(number_of_steps):
            self.simulate_step(1e-4)
            if vis:
                vis.Run()
                vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
                vis.Render()
                vis.EndScene()
        if visualize:
            vis.GetDevice().closeDevice()


@dataclass
class SimulationResult:
    """Data class to aggregate the output of the simulation.
    
        Attributes:
            time (float): the total simulation time
            time_vector (List[float]): the vector of time steps
            n_steps (int): the maximum possible number of steps
            robot_final_ds (Optional[DataStorage]): final data store of the robot
            environment_final_ds (Optional[DataStorage]): final data store of the environment"""
    time: float = 0
    time_vector: List[float] = field(default_factory=list)
    n_steps = 0
    robot_final_ds: Optional[DataStorage] = None
    environment_final_ds: Optional[DataStorage] = None

    def reduce_ending(self, step_n):
        if self.robot_final_ds:
            storage = self.robot_final_ds.main_storage
            for key in storage:
                key_storage = storage[key]
                for key_2 in key_storage:
                    value = key_storage[key_2]
                    new_value = value[:step_n + 2:]
                    key_storage[key_2] = new_value

        if self.environment_final_ds:
            storage = self.environment_final_ds.main_storage
            for key in storage:
                key_storage = storage[key]
                for key_2 in key_storage:
                    value = key_storage[key_2]
                    new_value = value[:step_n + 2:]
                    key_storage[key_2] = new_value

    def reduce_nan(self):
        if self.robot_final_ds:
            storage = self.robot_final_ds.main_storage
            for key in storage:
                key_storage = storage[key]
                for key_2 in key_storage:
                    value = key_storage[key_2]
                    new_value = [x for x in value if np.logical_not(np.isnan(x).all())]
                    key_storage[key_2] = new_value

        if self.environment_final_ds:
            storage = self.environment_final_ds.main_storage
            for key in storage:
                key_storage = storage[key]
                for key_2 in key_storage:
                    value = key_storage[key_2]
                    new_value = [x for x in value if np.logical_not(np.isnan(x).all())]
                    key_storage[key_2] = new_value


class RobotSimulationChrono():
    """The simulation of a robot within an environment.
    
        Attributes:
            chrono_system (chrono.ChSystem): the chrono simulation system that controls the 
                current simulation
            self.data (DataStorage): the object that aggregates the env_sensor data for the whole simulation
            env_sensor (Sensor): sensor attached to the environment
            objects : list of objects added to the environment
            active_body_counter: counter for environment bodies that added to the sensor
            active_objects : environment objects added to the env_sensor and env_data
            robot (RobotChrono): the robot added to the simulation
        """

    def __init__(self, object_list: List[Tuple[ChronoEasyShapeObject, bool]] = []):
        """Create a simulation system with some environment objects
        
            The robot and additional environment objects should be added using class methods.
            Args:
                object_list : bodies to add to the environment and their active/passive status"""
        # We assume that all simulations in one search are carried out with the same parameters that
        # can be set in the simulation constructor
        self.chrono_system = chrono.ChSystemNSC()
        self.chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.chrono_system.SetSolverMaxIterations(100)
        self.chrono_system.SetSolverForceTolerance(1e-6)
        self.chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0, 0, 0))
        # the simulating mechanism is to be added with function add_design, the value in constructor is None
        self.env_data_dict = {}
        self.robot_data_dict = {}
        self.result = SimulationResult()
        self.robot: Optional[RobotChrono] = None
        #self.env_sensor: Optional[Sensor] = None
        self.objects: List[ChronoEasyShapeObject] = []
        self.active_body_counter = 0
        self.active_objects_ordered: Dict[int, ChronoEasyShapeObject] = {}
        self.force_torque_container = ForceTorqueContainer()

        for obj in object_list:
            self.add_object(obj[0], obj[1])

    def add_env_data_type_dict(self, data_dict):
        self.env_data_dict = data_dict

    def add_robot_data_type_dict(self, data_dict):
        self.robot_data_dict = data_dict

    def initialize(self, max_number_of_steps) -> None:
        """Initialize Sensor for environment and data stores for robot and environment

            Args:
                max_number_of_steps (int): maximum number of steps in the simulation"""

        env_sensor: Sensor = Sensor(self.active_objects_ordered, {})
        env_sensor.contact_reporter.reset_contact_dict()
        self.data_storage: DataStorage = DataStorage(env_sensor)
        for key, value in self.env_data_dict.items():
            self.data_storage.add_data_type(key, value[0], value[1], max_number_of_steps)

        for key, value in self.robot_data_dict.items():
            self.robot.data_storage.add_data_type(key, value[0], value[1], max_number_of_steps)

    def add_design(self,
                   graph: GraphGrammar,
                   control_parameters,
                   control_cls=ConstController,
                   Frame: FrameTransform = DefaultFrame,
                   is_fixed=True,
                   with_data=True):
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
                                 [],is_fixed)
        self.robot_with_data = with_data

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

        self.chrono_system.AddBody(obj.body)
        self.objects.append(obj)
        if force_torque_controller:
            force_torque_controller.bind_body(obj.body)
            self.force_torque_container.add(force_torque_controller)
        if read_data:
            self.active_objects_ordered[self.active_body_counter] = obj
            self.active_body_counter += 1

    def update_data(self, step_n):
        """Update the env_sensor and env_data.
            Args:
                step_n (int): number of the current step"""
        self.data_storage.sensor.contact_reporter.reset_contact_dict()
        self.data_storage.sensor.update_current_contact_info(self.chrono_system)
        self.data_storage.update_storage(step_n)

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
        robot.controller.update_functions(current_time, robot.sensor, self.data_storage.sensor)
        self.force_torque_container.update_all(current_time, self.data_storage.sensor)

    def simulate(
        self,
        number_of_steps: int,
        step_length: float,
        frame_update: int,
        event_container: List[SimulationSingleEvent] = None,
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
        vis = None
        if visualize:
            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(self.chrono_system)
            vis.SetWindowSize(1024, 768)
            vis.SetWindowTitle('Grab demo')
            vis.Initialize()
            vis.AddCamera(chrono.ChVectorD(1.5, 3, -4))
            vis.AddTypicalLights()
            vis.EnableCollisionShapeDrawing(True)

        stop_flag = False
        self.result.time_vector = [0]
        for i in range(number_of_steps):
            current_time = self.chrono_system.GetChTime()
            self.simulate_step(step_length, current_time, i)
            self.result.time_vector.append(self.chrono_system.GetChTime(
            ))  # TODO: timevector can be constructed from number_of_steps and time_step
            if vis:
                vis.Run()
                if i % frame_update == 0:
                    vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
                    vis.Render()
                    vis.EndScene()
            if event_container:
                for event in event_container:
                    event_command = event.event_check(current_time, self.robot.sensor,
                                                      self.data_storage.sensor)
                    if event_command == EventCommands.STOP:
                        stop_flag = True
                        break

            if stop_flag:
                break

        if visualize:
            vis.GetDevice().closeDevice()

        self.result.environment_final_ds = self.data_storage
        self.result.robot_final_ds = self.robot.data_storage
        self.result.time = self.chrono_system.GetChTime()
        self.n_steps = number_of_steps
        self.result.reduce_nan()
        return self.result


class RobotSimulationWithForceTest(RobotSimulationChrono):

    def __init__(self, delay=False, object_list: List[Tuple[ChronoEasyShapeObject, bool]] = []):
        super().__init__(object_list)
        self.delay_flag = delay

    def activate(self, current_time):
        self.force_torque_container.controller_list[0].start_time = current_time

    def handle_single_events(self, event_container, current_time, step_n):
        if event_container is None:
            return False

        for event in event_container:
            if not event.state:
                event_command = event.event_check(current_time, step_n, self.robot.sensor,
                                                  self.data_storage.sensor)
                if event_command == EventCommands.STOP:
                    return True
                elif event_command == EventCommands.ACTIVATE:
                    self.activate(current_time)

        return False

    def simulate(
        self,
        number_of_steps: int,
        step_length: float,
        frame_update: int,
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
        vis = None
        if visualize:
            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(self.chrono_system)
            vis.SetWindowSize(1024, 768)
            vis.SetWindowTitle('Grab demo')
            vis.Initialize()
            vis.AddCamera(chrono.ChVectorD(1.5, 3, -4))
            vis.AddTypicalLights()
            # vis.EnableCollisionShapeDrawing(True)

        stop_flag = False
        self.result.time_vector = [0]
        FPS = 30
        frame_simulation = 0
        for i in range(number_of_steps):
            current_time = self.chrono_system.GetChTime()
            self.simulate_step(step_length, current_time, i)
            self.result.time_vector.append(self.chrono_system.GetChTime())
            if vis:
                vis.Run()
                if frame_simulation > 1/FPS/step_length:
                    frame_simulation = 0
                    vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
                    vis.Render()
                    vis.EndScene()
                    # just to slow down the simulation
                    if self.delay_flag:
                        time.sleep(0.0000001)
                else:
                    frame_simulation +=1

            stop_flag = self.handle_single_events(event_container, current_time, i)

            if stop_flag:
                break

        if visualize:
            vis.GetDevice().closeDevice()

        self.result.environment_final_ds = self.data_storage
        self.result.robot_final_ds = self.robot.data_storage
        self.result.time = self.chrono_system.GetChTime()
        self.n_steps = number_of_steps
        self.result.reduce_ending(i)
        return self.result