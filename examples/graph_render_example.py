 
from context import node
from node_render import *
mysystem = chrono.ChSystemNSC()

body_1 = ChronoBody(mysystem, length=1)
body_2 = ChronoBody(mysystem, length=0.5)
body_3 = ChronoBody(mysystem, length=0.5)
body_4 = ChronoBody(mysystem, length=0.5)
body_5 = ChronoBody(mysystem, length=1)
body_6 = ChronoBody(mysystem, length=1)
body_7 = ChronoBody(mysystem, length=1)

cord_sys_1 = chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.ChQuaternionD(1, 0, 0, 0))
transform1 = ChronoTransform(mysystem, cord_sys_1)

cord_sys_2 = chrono.ChCoordsysD(chrono.ChVectorD(0, 0.5, 0), chrono.Q_ROTATE_Z_TO_Y)
transform2 = ChronoTransform(mysystem, cord_sys_2)

cord_sys_3 = chrono.ChCoordsysD(chrono.ChVectorD(0, 0.0, 0), chrono.Q_ROTATE_Z_TO_Y)
transform3 = ChronoTransform(mysystem, cord_sys_3)

joint1 = ChronoRevolveJoint(mysystem)
joint2 = ChronoRevolveJoint(mysystem)
joint3 = ChronoRevolveJoint(mysystem)

# Fixed base
body_1.body.SetBodyFixed(True)

seq1 = [body_1, transform1, joint1, body_2, transform2, body_3]
seq2 = [body_2, transform3, joint2, body_4, transform3, joint3, body_5]
seq3 = [body_2, body_6, transform2, body_7]

connect_blocks(seq1)
connect_blocks(seq2)
connect_blocks(seq3)


vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(mysystem)
vis.SetWindowSize(1024,768)
vis.SetWindowTitle('Custom contact demo')
vis.Initialize()
 
 
vis.AddCamera(chrono.ChVectorD(8, 8, -6))
vis.AddTypicalLights()

 
 

while vis.Run():
    mysystem.Update()
    mysystem.DoStepDynamics(5e-3)
    vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
    vis.Render()
    
    vis.EndScene()