import pychrono as chrono
import pychrono.irrlicht as irr

from rostok import intexp

GENERATE_NEW_POSES = True

''' Testee object is dumbbell '''
obj_db = intexp.chrono_api.ChTesteeObject() # Create Chrono Testee Object

# Create 3D mesh and setup parameters from files
obj_db.create_chrono_body_from_file('./examples/models/custom/pipe.obj',
                                    './examples/models/custom/pipe.xml')

# Uncomment the line below for generate new poses
if GENERATE_NEW_POSES:
    # new_gen_poses = intexp.poses_generator.gen_random_poses_around_line(20, 0.04, 0.06, 0.015, 0.18)
    # new_gen_poses = intexp.poses_generator.gen_cylindrical_surface_from_poses(20, 0.07, 0.015, 0.18)
    new_gen_poses = intexp.poses_generator.gen_cylindrical_surface_around_object_axis(obj_db, 20, 0.07, 0.02, 'z')
    # new_gen_poses = intexp.poses_generator.gen_random_poses_around_object_axis(obj_db, 20, 0.07, 0.02, 'z')
    obj_db.rewrite_grasping_poses_list(new_gen_poses)

obj_db.demonstrate_object_and_grasping_poses()

desired_poses = obj_db.get_chrono_grasping_poses_list() # List of all grasping poses
hand_power_grasp_frame = chrono.ChFrameD(chrono.ChVectorD(0, 0, 0), chrono.Q_ROTATE_Z_TO_Y)  # Power grasp point in the ABS
obj_db.set_chrono_body_on_pose(desired_poses[0], hand_power_grasp_frame) # Position object for grasp pose into point

''' Floor added for clarity '''
floor = chrono.ChBodyEasyBox(1,0.005,1, 1000, True, False, chrono.ChMaterialSurfaceNSC())
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
vis.SetWindowTitle('Grasping positions generation example')
vis.SetSymbolScale(3)
vis.Initialize()
vis.AddLight(chrono.ChVectorD(0,10,0), 15)
vis.AddSkyBox()
vis.AddCamera(chrono.ChVectorD(1,0.5,1))
vis.EnableCollisionShapeDrawing(True)
vis.EnableBodyFrameDrawing(True)

counter = 0.0
i = 1

while vis.Run():
    system.Update()
    system.DoStepDynamics(1e-3)
    vis.BeginScene()
    vis.Render()
    vis.EndScene()

    if counter > 0.1: #Demonstration all poses every 0.1 seconds
        obj_db.set_chrono_body_on_pose(desired_poses[i], hand_power_grasp_frame)
        counter = 0.0
        i += 1
        if i >= len(desired_poses):
            i = 0
    else:
        counter += 1e-3
