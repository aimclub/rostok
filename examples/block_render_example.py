import pychrono.core as chrono
import pychrono.irrlicht as chronoirr

from rostok.block_builder.node_render import (ChronoRevolveJoint, ChronoTransform, LinkChronoBody,
                                              connect_blocks)
from rostok.block_builder.transform_srtucture import FrameTransform, OriginWorldFrame

# Create Chrono system instance
mysystem = chrono.ChSystemNSC()

# Init body blocks
body_1 = LinkChronoBody(mysystem, length_y=1)
body_2 = LinkChronoBody(mysystem, length_y=0.5)
body_3 = LinkChronoBody(mysystem, length_y=0.5)
body_4 = LinkChronoBody(mysystem, length_y=0.5)
body_5 = LinkChronoBody(mysystem, length_y=1)
body_6 = LinkChronoBody(mysystem, length_y=1)
body_7 = LinkChronoBody(mysystem, length_y=1)

# Init transforms
cord_sys_1 = OriginWorldFrame
transform1 = ChronoTransform(mysystem, cord_sys_1)

quat_z_y = chrono.Q_ROTATE_Z_TO_Y
cord_sys_2 = FrameTransform([0, 0.5, 0], [quat_z_y.e0, quat_z_y.e1, quat_z_y.e2, quat_z_y.e3])
transform2 = ChronoTransform(mysystem, cord_sys_2)

cord_sys_3 = FrameTransform([0, 0., 0], [quat_z_y.e0, quat_z_y.e1, quat_z_y.e2, quat_z_y.e3])
transform3 = ChronoTransform(mysystem, cord_sys_3)

# Init joints
joint1 = ChronoRevolveJoint(mysystem)
joint2 = ChronoRevolveJoint(mysystem)
joint3 = ChronoRevolveJoint(mysystem)

# Fixed base
body_1.body.SetBodyFixed(True)

# Set the sequence of blocks to be connected
# These blocks will be placed next to each other
seq1 = [body_1, transform1, joint1, body_2, transform2, body_3]
seq2 = [body_2, transform3, joint2, body_4, transform3, joint3, body_5]
seq3 = [body_2, body_6, transform2, body_7]

connect_blocks(seq1)
connect_blocks(seq2)
connect_blocks(seq3)

# Initialize chrono visualization system
vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(mysystem)
vis.SetWindowSize(1024, 768)
vis.SetWindowTitle('Custom contact demo')
vis.Initialize()

vis.AddCamera(chrono.ChVectorD(8, 8, -6))
vis.AddTypicalLights()

# Simulation loop
while vis.Run():
    mysystem.Update()
    mysystem.DoStepDynamics(5e-3)
    vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
    vis.Render()
    vis.EndScene()
