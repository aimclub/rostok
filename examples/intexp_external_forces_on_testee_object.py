import pychrono as chrono
import pychrono.irrlicht as irr
from rostok import intexp

""" Testee object is dumbbell """
obj_db = intexp.chrono_api.ChTesteeObject() # Create Chrono Testee Object
# create 3D mesh and parameters (grasping poses read from file)
obj_db.create_chrono_body_from_file('./examples/models/custom/body1.obj',
                                    './examples/models/custom/body1.xml')
obj_db.set_chrono_body_ref_frame_in_point(chrono.ChFrameD(chrono.ChVectorD(0,0.2,0),
                                                          chrono.ChQuaternionD(1,0,0,0)))

""" Impact on Testee Object """
impact = chrono.ChForce()   # Create Impact
obj_db.chrono_body.AddForce(impact) # Attach to ChBody

IMPACTS_M = [1, 1, 1, 2, 2, 2] # Exam with 6 constant forces and different directions
IMPACTS_DIR = [chrono.ChVectorD(1,0,0),
               chrono.ChVectorD(0,1,0),
               chrono.ChVectorD(0,0,1),
               chrono.ChVectorD(-1,0,0),
               chrono.ChVectorD(0,-1,0),
               chrono.ChVectorD(0,0,-1)]
IMPACTS_APPLICATION_POINT = [chrono.ChVectorD(0, 0, 0),
                             chrono.ChVectorD(0, 0, 0),
                             chrono.ChVectorD(0, 0, 0),
                             chrono.ChVectorD(0, 0, 0),
                             chrono.ChVectorD(0, 0, 0),
                             chrono.ChVectorD(0, 0, 0)]

# Switch timer of impact every 2 sec
tracking_timer = intexp.chrono_api.ImpactTimerParameters(test_bound=len(IMPACTS_DIR),
                                                         clock_bound=2.0, step = 1e-3)


''' Floor added for clarity '''
floor = chrono.ChBodyEasyBox(1,0.005,1, 1000, True, True, chrono.ChMaterialSurfaceNSC())
floor.SetPos(chrono.ChVectorD(0,-0.005,0))
floor.SetBodyFixed(True)
floor.SetName('Floor')
floor.GetVisualShape(0).SetColor(chrono.ChColor(80/255, 80/255, 80/255))

''' Simulation Solver '''
system = chrono.ChSystemNSC()
system.Set_G_acc(chrono.ChVectorD(0,0,0))
system.Add(obj_db.chrono_body) # Chrono Testee Object added to simulation
system.Add(floor)

''' PyChrono Visualisation '''
vis = irr.ChVisualSystemIrrlicht()
vis.AttachSystem(system)
vis.SetWindowSize(1028,768)
vis.SetWindowTitle('Impacts')
vis.SetSymbolScale(3)
vis.Initialize()
vis.AddLight(chrono.ChVectorD(0,10,0), 15)
vis.AddSkyBox()
vis.AddCamera(chrono.ChVectorD(1,0.5,1))
vis.EnableCollisionShapeDrawing(True)
vis.EnableBodyFrameDrawing(True)

while vis.Run():
    system.Update()
    system.DoStepDynamics(1e-3)
    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    # Update timer and go through the list of impacts
    intexp.chrono_api.update_impact(obj_db, impact, tracking_timer,
                                    IMPACTS_APPLICATION_POINT, IMPACTS_DIR, IMPACTS_M)
    