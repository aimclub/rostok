import context
from engine.node import GraphGrammar
from engine.node_render import ChronoBody
from utils.blocks_utils import make_collide, CollisionGroup   
import pychrono as chrono
import pychrono.irrlicht as chronoirr
from engine.robot import Robot
from chrono_simulatation import ChronoSimualtion
import engine.control as control
import time

# Function for stopping simulation by time optimize
class MinimizeStopper(object):
    def __init__(self, max_sec=0.3):
        self.max_sec = max_sec
        self.start = time.time()

    def __call__(self, xk=None, convergence=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            print("Terminating optimization: time limit reached")
            return True
        else:
            # you might want to report other stuff here
            # print("Elapsed: %.3f sec" % elapsed)
            return False

class SimulationStepOptimization:
    def __init__(self, control_trajectory, graph_mechanism: GraphGrammar, grasp_object: chrono.ChBody,  is_optimize: bool = False):
        self.control_trajectory = control_trajectory
        self.graph_mechanism = graph_mechanism
        self.grasp_object = grasp_object
        
        self.chrono_system = chrono.ChSystemNSC()
        self.grab_robot = Robot(self.graph_mechanism, self.chrono_system)
        
        ids_blocks = list(self.grab_robot.block_map.keys())
        base_id = graph_mechanism.closest_node_to_root(ids_blocks)
        self.grab_robot.block_map[base_id].body.SetBodyFixed(True)
        
        blocks = self.grab_robot.block_map.values()
        body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
        make_collide(body_block, CollisionGroup.Robot)
        
        self.chrono_system.Add(self.grasp_object)
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0,0,0))
        
        if is_optimize:
            # TODO: add function change joint to torque input
            pass
        self.controller_joints = []
        
        try:
            for id_finger, finger in enumerate(self.grasp_robot.get_joints):
                for id_joint, joint in finger:
                    self.controller_joints.append(control.TrackingControl(joint))
                    self.controller_joints[-1].set_des_positions(control_trajectory[id_finger][id_joint])
        except IndexError:
            raise IndexError("Arries control and joints aren't same shape")
    
    def change_config_system(self, dict_config: dict):
            for str_method, input in dict_config.items():
                try:
                    metod_system = getattr(self.chrono_system, str_method)
                    metod_system(input)
                except AttributeError:
                    raise AttributeError("Chrono system don't have method {0}".format(str_method))
                
    def simulate_system(self):
        
        
class SimulationLooper:
    def __init__(self,
                 chrono_system: chrono.ChSystem,
                 grab_robot: Robot,
                 grab_object: chrono.ChBody):
        self.chrono_system = chrono_system
        self.grab_robot = grab_robot
        self.grab_object = grab_object
        
    def set_function_constructor_object(self, function):
        self.function_constructor_object = function
    
    def set_function_constructor_system(self, function):
        self.function_constructor_system = function
        
    def set_function_constructor_robot(self, function):
        self.function_constructor_robot = function
        
    def initilize_looper(self, graph_mechanism: GraphGrammar):
        system: chrono.ChSystem = self.chrono_system()
        
        if hasattr(self,"function_constructor_system"):
            self.function_constructor_system(system)
        
        grab_robot: Robot = self.grab_robot(graph_mechanism, system)
        
        if hasattr(self,"function_constructor_robot"):
            self.function_constructor_robot(self.grab_robot)
        
        ids_blocks = list(grab_robot.block_map.keys())
        base_id = graph_mechanism.closest_node_to_root(ids_blocks)
        grab_robot.block_map[base_id].body.SetBodyFixed(True)
        
        if hasattr(self,"function_constructor_object"):
            self.function_constructor_object(self.grab_object)
            
        system.Add(self.grab_object)
        
        return system, grab_robot
        
        
    
    def do_iteration(self):
        pass
    
    
class ControlOptimizer:
    def __init__(self, graph_mechanism: GraphGrammar, num_iterations: int):
        self.graph_mechanism = graph_mechanism
        self.optimize_
        
        
    def __loop(self):
        mysystem = chrono.ChSystemNSC()
        mysystem.Set_G_acc(chrono.ChVectorD(0,0,0))

        # robot1 = robot.Robot(G, mysystem)
        # joint_blocks = robot1.get_joints

        # base_id = robot1.graph.find_nodes(F1)[0]
        # robot1.block_map[base_id].body.SetBodyFixed(True)

        # # Add fixed torque
        # controller = []
        # for joint in joint_blocks.values():
        #     controller.append(control.TrackingControl(joint))
        #     controller[-1].set_function_trajectory(lambda x: 1)

        # # Add object to grab
        # obj = chrono.ChBodyEasyBox(0.2,0.2,0.6,1000,True,True,mat)
        # obj.SetCollide(True)
        # obj.SetPos(chrono.ChVectorD(0,1.2,0))
        # mysystem.Add(obj)

        # # Make robot collide
        # blocks = robot1.block_map.values()
        # body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
        # make_collide(body_block, CollisionGroup.Robot)
        
        # Visualization
        vis = chronoirr.ChVisualSystemIrrlicht()
        vis.AttachSystem(mysystem)
        vis.SetWindowSize(1024,768)
        vis.SetWindowTitle('Grab demo')
        vis.Initialize()
        vis.AddCamera(chrono.ChVectorD(8, 8, -6))
        vis.AddTypicalLights()
        # print(i)
        while vis.Run():
            mysystem.Update()
            mysystem.DoStepDynamics(1e-3)
            vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
            vis.Render()
            vis.EndScene()
            if abs(mysystem.GetChTime() -  1) <= 0.001 :
                # mysystem.Clear()
                break