import pychrono.core as chrono
import pychrono.irrlicht as chronoirr

from rostok.block_builder_chrono.block_classes import (ChronoRevolveJoint, ChronoTransform,PrimitiveBody)
from rostok.block_builder_chrono.easy_body_shapes import Box
from rostok.block_builder_chrono.block_connect import place_and_connect                                            
from rostok.block_builder_chrono.blocks_utils import FrameTransform, OriginWorldFrame
from rostok.block_builder_chrono.chrono_system import register_chrono_system

# Create Chrono system instance
mysystem = chrono.ChSystemNSC()
register_chrono_system(mysystem)
# Init body blocks
body_1 = PrimitiveBody(Box(0.1, 1, 0.4))
body_2 = PrimitiveBody(Box(0.1, 0.5, 0.4))
body_3 = PrimitiveBody(Box(0.1, 0.5, 0.4))
body_4 = PrimitiveBody(Box(0.1, 0.5, 0.4))
body_5 = PrimitiveBody(Box(0.1, 1, 0.4))
body_6 = PrimitiveBody(Box(0.1, 1, 0.4))
body_7 = PrimitiveBody(Box(0.1, 1, 0.4))

# Init transforms
cord_sys_1 = OriginWorldFrame
transform1 = ChronoTransform( cord_sys_1)

quat_z_y = chrono.Q_ROTATE_Z_TO_Y
cord_sys_2 = FrameTransform([0, 0.5, 0], [quat_z_y.e0, quat_z_y.e1, quat_z_y.e2, quat_z_y.e3])
transform2 = ChronoTransform( cord_sys_2)

cord_sys_3 = FrameTransform([0, 0., 0], [quat_z_y.e0, quat_z_y.e1, quat_z_y.e2, quat_z_y.e3])
transform3 = ChronoTransform(cord_sys_3)

# Init joints
joint1 = ChronoRevolveJoint(starting_angle=10)
joint2 = ChronoRevolveJoint(starting_angle=15)
joint3 = ChronoRevolveJoint(starting_angle=20)

# Fixed base
body_1.body.SetBodyFixed(True)

# Set the sequence of blocks to be connected
# These blocks will be placed next to each other
seq1 = [body_1, transform1, joint1, body_2, transform2, body_3]
seq2 = [body_2, transform3, joint2, body_4, transform3, joint3, body_5]
seq3 = [body_2, body_6, transform2, body_7]

place_and_connect(seq1, mysystem)
place_and_connect(seq2, mysystem)
place_and_connect(seq3, mysystem)

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
