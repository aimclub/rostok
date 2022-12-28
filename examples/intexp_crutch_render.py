import open3d as o3d
import pychrono as chrono
import pychrono.irrlicht as irr
from numpy import asarray
from rostok.intexp.chrono_api import ChTesteeObject, ChCrutch

''' Floor added for clarity '''
floor = chrono.ChBodyEasyBox(1,0.005,1, 1000, True, True, chrono.ChMaterialSurfaceNSC())
floor.SetPos(chrono.ChVectorD(0,-0.1,0))
floor.SetBodyFixed(True)
floor.SetName('Floor')
floor.GetVisualShape(0).SetColor(chrono.ChColor(80/255, 80/255, 80/255))

''' Testee object placed in Crutch '''
obj_db = ChTesteeObject() # Create Chrono Testee Object
# Create 3D mesh and setup parameters from files
obj_db.create_chrono_body_from_file('./examples/models/custom/pipe.obj',
                                    './examples/models/custom/pipe.xml')
obj_db.set_chrono_body_ref_frame_in_point(chrono.ChFrameD(chrono.ChVectorD(0, 0.2, 0),
                                                          chrono.ChQuaternionD(1, 0, 0, 0)))

holder = ChCrutch(depth_k = 0.05)
holder.build_chrono_body(obj_db)

obj_db.set_chrono_body_ref_frame_in_point(holder.place_for_object)

system = chrono.ChSystemNSC()
system.Set_G_acc(chrono.ChVectorD(0,-9.8,0))
system.Add(obj_db.chrono_body) # Chrono Testee Object added to simulation
system.Add(floor)
system.Add(holder.chrono_body)
vis = irr.ChVisualSystemIrrlicht()
vis.AttachSystem(system)
vis.SetWindowSize(1028,768)
vis.SetWindowTitle('ZXC')
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
