from node import BlockWrapper, Node, Rule, Grammar
from node_render import *
from pychrono import ChCoordsysD, ChVectorD, ChQuaternionD
from pychrono import Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X, \
    Q_ROTATE_Y_TO_X, Q_ROTATE_Y_TO_Z, \
    Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z
import networkx as nx
import matplotlib.pyplot as plt
import pychrono as chrono
import control as ctrl
import numpy as np

# Define block types

# Bodies
link1 = BlockWrapper(ChronoBody, length=0.5)
link2 = BlockWrapper(ChronoBody, length=0.2)

flat1 = BlockWrapper(ChronoBody, width=0.4, length=0.1)
flat2 = BlockWrapper(ChronoBody, width=0.7, length=0.1)

u1 = BlockWrapper(ChronoBody, width=0.1, length=0.1)

# Transforms
RZX = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_X)
RZY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_Y)
RXY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_X_TO_Y)

MOVE_ZX_PLUS = ChCoordsysD(ChVectorD(0.3, 0, 0.3), ChQuaternionD(1, 0, 0, 0))
MOVE_ZX_MINUS = ChCoordsysD(ChVectorD(-0.3, 0, -0.3), ChQuaternionD(1, 0, 0, 0))

transform_rzx = BlockWrapper(ChronoTransform, RZX)
transform_rzy = BlockWrapper(ChronoTransform, RZY)
transform_rxy = BlockWrapper(ChronoTransform, RXY)
transform_mzx_plus = BlockWrapper(ChronoTransform, MOVE_ZX_PLUS)
transform_mzx_minus = BlockWrapper(ChronoTransform, MOVE_ZX_MINUS)

type_of_input = ChronoRevolveJoint.InputType.Velocity
# Joints
revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)

# Defines rules
# Nodes
ROOT = Node("ROOT")

J1 = Node(label="J1", is_terminal=True, block_wrapper=revolve1)
L1 = Node(label="L1", is_terminal=True, block_wrapper=link1)
L2 = Node(label="L2", is_terminal=True, block_wrapper=link2)
F1 = Node(label="F1", is_terminal=True, block_wrapper=flat1)
F2 = Node(label="F2", is_terminal=True, block_wrapper=flat2)
U1 = Node(label="U1", is_terminal=True, block_wrapper=u1)
T1 = Node(label="T1", is_terminal=True, block_wrapper=transform_rzx)
T2 = Node(label="T2", is_terminal=True, block_wrapper=transform_rzy)
T3 = Node(label="T3", is_terminal=True, block_wrapper=transform_mzx_plus)
T4 = Node(label="T4", is_terminal=True, block_wrapper=transform_mzx_minus)

J = Node("J")
L = Node("L")
F = Node("F")
M = Node("M")
EF = Node("EF")
EM = Node("EM")

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


G = Grammar()

rule_action = [FlatCreate, Mount, Mount, FingerUpper, FingerUpper,
               TerminalFlat, TerminalL1, TerminalL1, TerminalTransformR, TerminalTransformL, TerminalEndLimb,
               TerminalEndLimb,
               TerminalJoint, TerminalJoint]

for i in rule_action:
    G.apply_rule(i)

mysystem = chrono.ChSystemNSC()
wrapper_array = G.build_wrapper_array()


blocks = []
uniq_blocks = {}
for wrap in wrapper_array:
    block_line = []
    for id, wrapper in wrap:
        if not (id in uniq_blocks.keys()):
            wrapper.builder = mysystem
            block_buf = wrapper.create_block()
            block_line.append(block_buf)
            uniq_blocks[id] = block_buf
        else:
            block_buf = uniq_blocks[id]
            block_line.append(block_buf)
    blocks.append(block_line)

for line in blocks:
    build_branch(line)
blocks[0][0].body.SetBodyFixed(True)

# Create simulation loop
des_points_1 = 1*np.array([0, -0.3, 0.3, -0.2, 0.4])
des_points_2 = -1*np.array(des_points_1)

track_ctrl_j_1 = ctrl.TrackingControl(blocks[0][2], des_points_1, (0.5,1.5))
track_ctrl_j_2 = ctrl.TrackingControl(blocks[1][2], des_points_2, (0.5,1.5))

vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(mysystem)
vis.SetWindowSize(1024,768)
vis.SetWindowTitle('Custom contact demo')
vis.Initialize()
vis.AddCamera(chrono.ChVectorD(8, 8, -6))
vis.AddTypicalLights()

plt.figure()
nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800,
                 labels={n: G.nodes[n]["Node"].label for n in G})
plt.figure()
nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800)
plt.show()

while vis.Run():
    mysystem.Update()
    mysystem.DoStepDynamics(5e-3)
    vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
    vis.Render()
    
    vis.EndScene()
