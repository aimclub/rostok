
import context
import networkx as nx
import numpy as np
import pychrono as chrono
import pychrono.irrlicht as chronoirr
from pychrono import (Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z, Q_ROTATE_Y_TO_X,
                      Q_ROTATE_Y_TO_Z, Q_ROTATE_Z_TO_X, Q_ROTATE_Z_TO_Y,
                      ChCoordsysD, ChQuaternionD, ChVectorD)

import engine.control as ctrl
import engine.robot as robot
from engine.node import ROOT, BlockWrapper, GraphGrammar, Node, Rule
from engine.node_render import ChronoBody, ChronoRevolveJoint, ChronoTransform
from utils.blocks_utils import CollisionGroup, make_collide
from utils.flags_simualtions import FlagMaxTime
from utils.dataset_materials.material_dataclass_manipulating import create_struct_material_from_file
from utils.transform_srtucture import FrameTransform

# Define block types
mat = chrono.ChMaterialSurfaceNSC()
mat.SetFriction(0.5)
mat.SetDampingF(0.1)

mat_r = ("polyactide", "./src/utils/dataset_materials/material.xml", "ChMaterialSurfaceNSC")
polyactide_material_struct = create_struct_material_from_file(*mat_r)
# Bodies
link1 = BlockWrapper(ChronoBody, length=0.6, material = polyactide_material_struct)
link2 = BlockWrapper(ChronoBody, length=0.4, material = polyactide_material_struct)

flat1 = BlockWrapper(ChronoBody, width=0.8, length=0.2)
flat2 = BlockWrapper(ChronoBody, width=1.4, length=0.2)

u1 = BlockWrapper(ChronoBody, width=0.2, length=0.2)

# Transforms

MOVE_ZX_PLUS = FrameTransform([0.3,0,0.3],[1,0,0,0])
MOVE_ZX_MINUS = FrameTransform([-0.3,0,-0.3],[1,0,0,0])

MOVE_X_PLUS = FrameTransform([0.3,0,0.],[1,0,0,0])
MOVE_Z_PLUS_X_MINUS = FrameTransform([-0.3,0,0.3],[1,0,0,0])

transform_mzx_plus = BlockWrapper(ChronoTransform, MOVE_ZX_PLUS)
transform_mzx_minus = BlockWrapper(ChronoTransform, MOVE_ZX_MINUS)
transform_mx_plus = BlockWrapper(ChronoTransform, MOVE_X_PLUS)
transform_mz_plus_x_minus = BlockWrapper(ChronoTransform, MOVE_Z_PLUS_X_MINUS)

type_of_input = ChronoRevolveJoint.InputType.Torque
# Joints
revolve1 = BlockWrapper(
    ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)


J1 = Node(label="J1", is_terminal=True, block_wrapper=revolve1)
L1 = Node(label="L1", is_terminal=True, block_wrapper=link1)
L2 = Node(label="L2", is_terminal=True, block_wrapper=link2)
F1 = Node(label="F1", is_terminal=True, block_wrapper=flat1)
F2 = Node(label="F2", is_terminal=True, block_wrapper=flat2)
U1 = Node(label="U1", is_terminal=True, block_wrapper=u1)
T1 = Node(label="T1", is_terminal=True, block_wrapper=transform_mx_plus)
T2 = Node(label="T2", is_terminal=True,
          block_wrapper=transform_mz_plus_x_minus)
T3 = Node(label="T3", is_terminal=True, block_wrapper=transform_mzx_plus)
T4 = Node(label="T4", is_terminal=True, block_wrapper=transform_mzx_minus)


J = Node("J")
L = Node("L")
F = Node("F")
M = Node("M")
EF = Node("EF")
EM = Node("EM")

# Defines rules

# Non terminal

# Simple replace
FlatCreate = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=F)
FlatCreate.id_node_connect_child = 0
FlatCreate.id_node_connect_parent = 0
FlatCreate.graph_insert = rule_graph
FlatCreate.replaced_node = ROOT

Mount = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=F)
rule_graph.add_node(1, Node=M)
rule_graph.add_node(2, Node=EM)
rule_graph.add_edge(0, 1)
rule_graph.add_edge(1, 2)
Mount.id_node_connect_child = 0
Mount.id_node_connect_parent = 2
Mount.graph_insert = rule_graph
Mount.replaced_node = F

MountAdd = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=M)
rule_graph.add_node(1, Node=EM)
rule_graph.add_edge(0, 1)
MountAdd.id_node_connect_child = 1
MountAdd.id_node_connect_parent = 0
MountAdd.graph_insert = rule_graph
MountAdd.replaced_node = M

FingerUpper = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=J)
rule_graph.add_node(1, Node=L)
rule_graph.add_node(2, Node=EM)
rule_graph.add_edge(0, 1)
rule_graph.add_edge(1, 2)
FingerUpper.id_node_connect_child = 2
FingerUpper.id_node_connect_parent = 0
FingerUpper.graph_insert = rule_graph
FingerUpper.replaced_node = EM

# Terminal
TerminalFlat = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=F1)
TerminalFlat.id_node_connect_child = 0
TerminalFlat.id_node_connect_parent = 0
TerminalFlat.graph_insert = rule_graph
TerminalFlat.replaced_node = F

TerminalL1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=L1)
TerminalL1.id_node_connect_child = 0
TerminalL1.id_node_connect_parent = 0
TerminalL1.graph_insert = rule_graph
TerminalL1.replaced_node = L

TerminalL2 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=L2)
TerminalL2.id_node_connect_child = 0
TerminalL2.id_node_connect_parent = 0
TerminalL2.graph_insert = rule_graph
TerminalL2.replaced_node = L

TerminalTransformRX = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=T1)
TerminalTransformRX.id_node_connect_child = 0
TerminalTransformRX.id_node_connect_parent = 0
TerminalTransformRX.graph_insert = rule_graph
TerminalTransformRX.replaced_node = M

TerminalTransformLZ = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=T2)
TerminalTransformLZ.id_node_connect_child = 0
TerminalTransformLZ.id_node_connect_parent = 0
TerminalTransformLZ.graph_insert = rule_graph
TerminalTransformLZ.replaced_node = M

TerminalTransformR = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=T3)
TerminalTransformR.id_node_connect_child = 0
TerminalTransformR.id_node_connect_parent = 0
TerminalTransformR.graph_insert = rule_graph
TerminalTransformR.replaced_node = M

TerminalTransformL = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=T4)
TerminalTransformL.id_node_connect_child = 0
TerminalTransformL.id_node_connect_parent = 0
TerminalTransformL.graph_insert = rule_graph
TerminalTransformL.replaced_node = M

TerminalEndLimb = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=U1)
TerminalEndLimb.id_node_connect_child = 0
TerminalEndLimb.id_node_connect_parent = 0
TerminalEndLimb.graph_insert = rule_graph
TerminalEndLimb.replaced_node = EM

TerminalJoint = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=J1)
TerminalJoint.id_node_connect_child = 0
TerminalJoint.id_node_connect_parent = 0
TerminalJoint.graph_insert = rule_graph
TerminalJoint.replaced_node = J


G = GraphGrammar()

rule_action_non_terminal = np.asarray([FlatCreate, Mount, Mount, Mount,
                                       FingerUpper, FingerUpper, FingerUpper, FingerUpper,  FingerUpper, FingerUpper])
rule_action_terminal = np.asarray([TerminalFlat,
                                   TerminalL1, TerminalL1, TerminalL1, TerminalL2, TerminalL2, TerminalL2,
                                   TerminalTransformL, TerminalTransformLZ, TerminalTransformRX,
                                   TerminalEndLimb, TerminalEndLimb, TerminalEndLimb,
                                   TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint])
rule_action = np.r_[rule_action_non_terminal, rule_action_terminal]
for i in list(rule_action):
    G.apply_rule(i)

mysystem = chrono.ChSystemNSC()
mysystem.Set_G_acc(chrono.ChVectorD(0, 0, 0))

grab_robot = robot.Robot(G, mysystem)
joint_blocks = grab_robot.get_joints

obj = chrono.ChBodyEasyBox(0.2, 0.2, 0.6, 1000, True, True, mat)
obj.SetCollide(True)
obj.SetPos(chrono.ChVectorD(0, 1.2, 0))
mysystem.Add(obj)

base_id = grab_robot.get_block_graph().find_nodes(F1)[0]
grab_robot.block_map[base_id].body.SetBodyFixed(True)


des_points_1 = np.array([0, 0.1, 0.2, 0.3, 0.4])*2
des_points_1_1 = - des_points_1

pid_track = []
for id, finger in enumerate(joint_blocks):
    for joint in finger:
        if id != 2:
            pid_track.append(ctrl.ChControllerPID(joint, 80., 5., 1.))
            pid_track[-1].set_des_positions_interval(des_points_1, (0.1, 2))
        else:
            pid_track.append(ctrl.ChControllerPID(joint, 80., 5., 1.))
            pid_track[-1].set_des_positions_interval(des_points_1_1, (0.1, 2))

# Visualization
# plot_graph(G)

vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(mysystem)
vis.SetWindowSize(1024, 768)
vis.SetWindowTitle('Custom contact demo')
vis.Initialize()
vis.AddCamera(chrono.ChVectorD(8, 8, -6))
vis.AddTypicalLights()


# Make robot collide
blocks = grab_robot.block_map.values()
body_block = filter(lambda x: isinstance(x, ChronoBody), blocks)
make_collide(body_block, CollisionGroup.Robot)


stoper = FlagMaxTime(10)
stoper.build(mysystem, grab_robot, obj)

# Create simulation loop
while vis.Run() and not stoper.get_flag_state():
    mysystem.Update()
    mysystem.DoStepDynamics(5e-3)
    vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
    vis.Render()
    vis.EndScene()
