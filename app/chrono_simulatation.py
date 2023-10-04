import pychrono as chrono
import pychrono.irrlicht as chronoirr

class ChronoSimualtion:
    def __init__(self, chrono_system: chrono.ChSystem, time_step = 1e-3,
                 sky_color: chrono.ChColor = chrono.ChColor(0.2, 0.2, 0.3), visualize = False) -> None:
        self.visualize = visualize
        self.step_time = time_step
        self.stop_time = float('inf')
        self.color_scenes = sky_color
        self.output = None
        self.__system = chrono_system
        
        if self.visualize:
            self.__vis_irrlicht = chronoirr.ChVisualSystemIrrlicht()
            self.__vis_irrlicht.AttachSystem(self.__system)
            
    
    def visualizer(self) -> chronoirr.ChVisualSystemIrrlicht:
        return self.__vis_irrlicht
    
    def __visual_simulation(self):
        while self.__vis_irrlicht.Run():
            self.__system.Update()
            self.__system.DoStepDynamics(self.step_time)
            self.__vis_irrlicht.BeginScene(True, True, self.color_scenes)
            self.__vis_irrlicht.Render()
            self.__vis_irrlicht.EndScene()
            if self.output:
                self.output()
            if self.__system.GetChTime() > self.stop_time:
                break
    
    def __non_visual_simulation(self):
        while (self.__system.GetChTime() <= self.stop_time):
            self.__system.DoStepDynamics(self.step_time)
            if self.output:
                self.output()
    
    def run_simulation(self):
        if self.visualize:
            self.__visual_simulation()
        else:
            self.__non_visual_simulation()
                  
    def output_terminal(self, output_function):
        self.output = output_function
        
    def standart_property_visilize(self):
        try:
            self.__vis_irrlicht.SetWindowSize(1080,720)
            self.__vis_irrlicht.SetWindowTitle('Visualization simulation')
            self.__vis_irrlicht.Initialize()
            self.__vis_irrlicht.AddCamera(chrono.ChVectorD(1, 1, -2))
            self.__vis_irrlicht.AddTypicalLights()
        except AttributeError:
            raise Exception("Simulate without visualization")