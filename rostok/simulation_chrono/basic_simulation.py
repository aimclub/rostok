from typing import List, Optional, Tuple

import pychrono as chrono
import pychrono.irrlicht as chronoirr

from rostok.block_builder_api.block_parameters import (DefaultFrame,
                                                       FrameTransform)
from rostok.block_builder_chrono.block_classes import ChronoEasyShapeObject
from rostok.virtual_experiment.robot_new import BuiltGraphChrono, RobotChrono
from rostok.virtual_experiment.sensors import ContactReporter, Sensor
from rostok.graph_grammar.node import GraphGrammar


class SystemPreviewChrono:
    """A simulation of the motionless environment and design"""
    def __init__(self):
        """Initialize the chroon system with default parameters"""
        self.chrono_system = chrono.ChSystemNSC()
        self.chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.chrono_system.SetSolverMaxIterations(100)
        self.chrono_system.SetSolverForceTolerance(1e-6)
        self.chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0, 0, 0))

    def add_design(self, graph, frame: FrameTransform = DefaultFrame):
        """Add a design into the system

            Args:
                graph (GraphGrammar): graph of the design
                frame (FrameTransform): initial position of the base body
        """

        BuiltGraphChrono(graph, self.chrono_system, frame, True)

    def add_object(self, obj: ChronoEasyShapeObject):
        """Add an object to the environment.

            Args:
                obj (ChronoEasyShapeObject): one of the simple chrono objects"""
        self.chrono_system.AddBody(obj.body)

    def simulate_step(self, time_step: float):
        """Simulate one step"""
        self.chrono_system.Update()
        self.chrono_system.DoStepDynamics(time_step)
        # TODO: add some check for collisions that can reveal the errors in objects or design positions.

    def simulate(self,
                 number_of_steps: int,
                 visualize:bool = True):
        """Simulate several steps and visualize system.

            The simulation purpose is to check the initial positions of objects and visualize the 
                environment and the mech design. More steps == longer simulation.

            Args:
                number_of_steps (int): the number of steps for simulation
                visualize (bool): the flag for visualization
        """
        #TODO: try to replace the number_of_steps for the stop by a button.
        if visualize:
            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(self.chrono_system)
            vis.SetWindowSize(1024, 768)
            vis.SetWindowTitle('Grab demo')
            vis.Initialize()
            vis.AddCamera(chrono.ChVectorD(1.5, 3, -2))
            vis.AddTypicalLights()
            vis.EnableCollisionShapeDrawing(True)

        for _ in range(number_of_steps):
            self.simulate_step(10e-3)
            if vis:
                vis.Run()
                vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
                vis.Render()
                vis.EndScene()
        if visualize:
            vis.GetDevice().closeDevice()


class RobotSimulationChrono():
    """The simulation of a robot within an environment.
    
        Attributes:
            chrono_system (chrono.ChSystem): the chrono simulation system that controls the current simulation
            self.data : the object for final output of the simulation
            env_sensor (Sensor): sensor attached to the environment
            objects : list of objects added to the environment
            active_body_counter: counter for environment bodies that added to the sensor
            active_objects : environment objects added to the env_sensor
            robot : the robot added to the simulation
        """
    def __init__(self,
                 object_list: List[Tuple[ChronoEasyShapeObject, bool]]):
        """Create a simulation system with some environment objects
        
            The robot and additional environment objects should be added using class methods.
            object_list : bodies to add to the environment and their active/passive status"""
        self.chrono_system = chrono.ChSystemNSC()
        self.chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.chrono_system.SetSolverMaxIterations(100)
        self.chrono_system.SetSolverForceTolerance(1e-6)
        self.chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0, 0, 0))

        self.data = None
        self.robot:Optional[RobotChrono] = None
        self.env_sensor: Sensor = Sensor([], {})
        self.objects: List[ChronoEasyShapeObject] = []
        self.active_body_counter = 0
        self.active_objects: List[Tuple[int, ChronoEasyShapeObject]] = []
        for obj in object_list:
            self.add_object(obj[0], obj[1])

    def initialize(self):
        pass

    def add_design(self, graph, control_parameters, control_trajectories=None,Frame: FrameTransform = DefaultFrame):
        """"""
        self.robot = RobotChrono(graph, self.chrono_system, control_parameters,control_trajectories ,Frame)

    def add_object(self, obj: ChronoEasyShapeObject, read_data: bool = False):
        self.chrono_system.AddBody(obj.body)
        self.objects.append(obj)
        if read_data:
            self.active_objects.append((self.active_body_counter, obj))
            self.active_body_counter += 1
            self.env_sensor.contact_reporter.set_body_list(self.active_objects)

    def update_data(self):
        self.env_sensor.contact_reporter.reset_contact_dict()
        self.env_sensor.update_current_contact_info(self.chrono_system)

    def get_current_data(self):
        None

    def simulate_step(self, time_step: float, current_time):
        self.chrono_system.Update()
        self.chrono_system.DoStepDynamics(time_step)
        self.update_data()

        robot:Robot = self.robot
        robot.sensor.contact_reporter.reset_contact_dict()
        robot.sensor.update_current_contact_info(self.chrono_system)
        robot.sensor.update_trajectories()
        robot.controller.update_functions(current_time, robot.sensor, self.get_current_data())

    def simulate(self,
                 number_of_steps: int,
                 step_length: float,
                 frame_update: int,
                 visualize=False):
        if visualize:
            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(self.chrono_system)
            vis.SetWindowSize(1024, 768)
            vis.SetWindowTitle('Grab demo')
            vis.Initialize()
            vis.AddCamera(chrono.ChVectorD(1.5, 3, -4))
            vis.AddTypicalLights()
            vis.EnableCollisionShapeDrawing(True)

        for i in range(number_of_steps):
            self.simulate_step(step_length, self.chrono_system.GetChTime())
            if vis:
                vis.Run()
                if i % frame_update == 0:
                    vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
                    vis.Render()
                    vis.EndScene()
        if visualize:
            vis.GetDevice().closeDevice()

        return self.data
