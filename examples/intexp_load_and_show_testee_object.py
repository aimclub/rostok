import pychrono as chrono
import pychrono.irrlicht as irr

from rostok import intexp

''' Testee object is dumbbell '''
obj_db = intexp.chrono_api.ChTesteeObject() # Create Chrono Testee Object
error = obj_db.createChronoBodyMeshFromFile('./examples/models/custom/body1.obj',
                                            './examples/models/custom/body1.xml') # create 3D mesh and parameters (grasping poses read from file)
print('Uploded mesh with code error:', error)
# Uncomment the line below for generate new poses
# new_gen_poses = intexp.poses_generator.genRandomPosesAroundLine(20, 0.04, 0.06, 0.015, 0.18)
# new_gen_poses = intexp.poses_generator.genCylindricalSurfaceFromPoses(20, 0.07, 0.015, 0.18)
# new_gen_poses = intexp.poses_generator.genRandomPosesAroundTesteeObjectAxis(obj_db, 20, 0.07, 0.02, 'y')
new_gen_poses = intexp.poses_generator.genCylindricalSurfaceAroundTesteeObjectAxis(obj_db, 20, 0.07, 0.02, 'y')
if new_gen_poses is not None:
    obj_db.setGraspingPosesList(new_gen_poses)
else:
    print('ERROR')
    
obj_db.showObjectWithGraspingPoses()
desired_poses = obj_db.getChronoGraspingPosesList() # List of all grasping poses
hand_power_grasp_frame = chrono.ChFrameD(chrono.ChVectorD(0, 0, 0), chrono.Q_ROTATE_Z_TO_Y)  # Power grasp point in the ABS
obj_db.setChronoBodyMeshOnPose(desired_poses[0], hand_power_grasp_frame) # Position object for grasp pose into point

''' Floor added for clarity '''
floor = chrono.ChBodyEasyBox(1,0.005,1, 1000, True, False, chrono.ChMaterialSurfaceNSC())
floor.SetPos(chrono.ChVectorD(0,-0.005,0))
floor.SetBodyFixed(True)
floor.SetName('Floor')
floor.GetVisualShape(0).SetColor(chrono.ChColor(80/255, 80/255, 80/255))

''' Simulation Solver '''
system = chrono.ChSystemNSC()
system.Set_G_acc(chrono.ChVectorD(0,0,0))
system.Add(obj_db.chrono_body_mesh) # Chrono Testee Object added to simulation
system.Add(floor)
''' PyChrono Visualisation '''
vis = irr.ChVisualSystemIrrlicht()
vis.AttachSystem(system)
vis.SetWindowSize(1028,768)
vis.SetWindowTitle('Grasping positions generation example')
vis.SetSymbolScale(3)
vis.Initialize()
vis.AddLight(chrono.ChVectorD(0,10,0), 15)
vis.AddSkyBox()
vis.AddCamera(chrono.ChVectorD(1,0.5,1))
vis.EnableCollisionShapeDrawing(True)
vis.EnableBodyFrameDrawing(True)

counter = 0
i = 1

while vis.Run():
    system.Update()
    system.DoStepDynamics(1e-3)
    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    
    if counter > 0.1: #Demonstration all poses every 0.5 seconds
        obj_db.setChronoBodyMeshOnPose(desired_poses[i], hand_power_grasp_frame)
        counter = 0
        i += 1
        if i >= len(desired_poses):
            i = 0
    else:    
        counter += 1e-3