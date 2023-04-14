import pychrono as chrono
import pychrono.irrlicht as chronoirr

from rostok.virtual_experiment.robot import BuiltGraph
from rostok.block_builder_api.block_parameters import FrameTransform, DefaultFrame

class BasicSimulation:
    def __init__(self, visualize:bool = True):
        self.chrono_system = chrono.ChSystemNSC()
        self.chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.chrono_system.SetSolverMaxIterations(100)
        self.chrono_system.SetSolverForceTolerance(1e-6)
        self.chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0, 0, 0))
        self.vis = None
        if visualize:
            # 30 fps
            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(self.chrono_system)
            vis.SetWindowSize(1024, 768)
            vis.SetWindowTitle('Grab demo')
            vis.Initialize()
            vis.AddCamera(chrono.ChVectorD(1.5, 3, -2))
            vis.AddTypicalLights()
            vis.EnableCollisionShapeDrawing(True)
            self.vis = vis

    def add_design(self, graph, Frame:FrameTransform = DefaultFrame):
        BuiltGraph(graph, self.chrono_system, True, Frame)

    def simulate_step(self,  time_step:float):
        self.chrono_system.Update()
        self.chrono_system.DoStepDynamics(time_step)

    def simulate(self, number_of_steps:int, step_length:float, frame_update:int):
        for i in range(number_of_steps):
            self.simulate_step(step_length)
            if self.vis: 
                self.vis.Run()
                if i%frame_update==0:
                    self.vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
                    self.vis.Render()
                    self.vis.EndScene()
