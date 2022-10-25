import context
from engine.node import GraphGrammar
import pychrono as chrono
import pychrono.irrlicht as chronoirr
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

class SimulationLooper:
    def __init__(self):
        pass
    
    def do_iteration(self):
        pass
    
    
class ControlOptimizer:
    def __init__(self, graph_mechanism: GraphGrammar, num_iterations: int):
        self.graph_mechanism = graph_mechanism
        self.optimize_
        
        
    def __loop(self):
        mysystem = chrono.ChSystemNSC()
        mysystem.Set_G_acc(chrono.ChVectorD(0,0,0))

        robot1 = robot.Robot(G, mysystem)
        joint_blocks = robot1.get_joints

        base_id = robot1.graph.find_nodes(F1)[0]
        robot1.block_map[base_id].body.SetBodyFixed(True)

        # Add fixed torque
        controller = []
        for joint in joint_blocks.values():
            controller.append(control.TrackingControl(joint))
            controller[-1].set_function_trajectory(lambda x: 1)

        # Add object to grab
        obj = chrono.ChBodyEasyBox(0.2,0.2,0.6,1000,True,True,mat)
        obj.SetCollide(True)
        obj.SetPos(chrono.ChVectorD(0,1.2,0))
        mysystem.Add(obj)

        # Make robot collide
        blocks = robot1.block_map.values()
        body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
        make_collide(body_block, CollisionGroup.Robot)
        
        # Visualization
        vis = chronoirr.ChVisualSystemIrrlicht()
        vis.AttachSystem(mysystem)
        vis.SetWindowSize(1024,768)
        vis.SetWindowTitle('Grab demo')
        vis.Initialize()
        vis.AddCamera(chrono.ChVectorD(8, 8, -6))
        vis.AddTypicalLights()
        print(i)
        while vis.Run():
            mysystem.Update()
            mysystem.DoStepDynamics(1e-3)
            vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
            vis.Render()
            vis.EndScene()
            if abs(mysystem.GetChTime() -  1) <= 0.001 :
                # mysystem.Clear()
                break