from typing import List, Tuple

import pychrono as chrono
import pychrono.irrlicht as chronoirr

from rostok.block_builder_api.block_parameters import (DefaultFrame, FrameTransform)
from rostok.block_builder_chrono.block_classes import ChronoEasyShapeObject
from rostok.virtual_experiment.robot import BuiltGraph, Robot
from rostok.virtual_experiment.sensors import ContactReporter


class SystemPreview:

    def __init__(self):
        self.chrono_system = chrono.ChSystemNSC()
        self.chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.chrono_system.SetSolverMaxIterations(100)
        self.chrono_system.SetSolverForceTolerance(1e-6)
        self.chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0, 0, 0))

    def add_design(self, graph, frame: FrameTransform = DefaultFrame):
        BuiltGraph(graph, self.chrono_system, True, frame)

    def simulate_step(self, time_step: float):
        self.chrono_system.Update()
        self.chrono_system.DoStepDynamics(time_step)

    def add_object(self, obj: ChronoEasyShapeObject):
        self.chrono_system.AddBody(obj.body)

    def simulate(self,
                 number_of_steps: int,
                 visualize=False):
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

    def __init__(self,
                 object_list: List[Tuple[ChronoEasyShapeObject, bool]], 
                 visualize: bool = True):
        super().__init__(visualize)
        self.data = None
        self.robot_list:List[Robot] = []
        self.env_reporter = ContactReporter()
        self.objects: List[ChronoEasyShapeObject] = []
        self.active_body_counter = 0
        self.active_objects: List[Tuple[int, ChronoEasyShapeObject]] = []
        for obj in object_list:
            self.add_object(obj[0], obj[1])

    def initialize(self):
        pass

    def add_design(self, graph, control_parameters, Frame: FrameTransform = DefaultFrame):
        self.robot_list.append(Robot(graph, self.chrono_system, Frame))

    def add_object(self, obj: ChronoEasyShapeObject, read_data: bool = False):
        self.chrono_system.AddBody(obj.body)
        self.objects.append(obj)
        if read_data:
            self.active_objects.append((self.active_body_counter, obj))
            self.active_body_counter += 1
            self.env_reporter.set_body_list(self.active_objects)

    def update_data(self):
        pass

    def simulate_step(self, time_step: float):
        self.chrono_system.Update()
        self.chrono_system.DoStepDynamics(time_step)
        self.env_reporter.collect_current_contacts(self.chrono_system)
        for robot in self.robot_list:
            robot.contact_reporter.collect_current_contacts(self.chrono_system)

        for robot in self.robot_list:
            robot.controller.update_functions()

        self.update_data()

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
            vis.AddCamera(chrono.ChVectorD(1.5, 3, -2))
            vis.AddTypicalLights()
            vis.EnableCollisionShapeDrawing(True)

        for i in range(number_of_steps):
            self.simulate_step(step_length)
            if vis:
                vis.Run()
                if i % frame_update == 0:
                    vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
                    vis.Render()
                    vis.EndScene()
        if visualize:
            vis.GetDevice().closeDevice()

        return self.data
