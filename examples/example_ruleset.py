import context
import pychrono as chrono
import networkx as nx
import numpy as np

from engine.node import BlockWrapper, Node, Rule, GraphGrammar, ROOT
from engine.node_render import ChronoBody, ChronoTransform, ChronoRevolveJoint

from utils.blocks_utils import make_collide, CollisionGroup
from pychrono import ChCoordsysD, ChVectorD, ChQuaternionD
from pychrono import Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X, \
    Q_ROTATE_Y_TO_X, Q_ROTATE_Y_TO_Z, \
    Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z


# Define block types
mat = chrono.ChMaterialSurfaceNSC()
mat.SetFriction(0.5)
mat.SetDampingF(0.1)

# Bodies
link1 = BlockWrapper(ChronoBody, length=0.3, material=mat)
link2 = BlockWrapper(ChronoBody, length=0.2, material=mat)

flat1 = BlockWrapper(ChronoBody, width=0.4, length=0.1, material=mat)
flat2 = BlockWrapper(ChronoBody, width=0.7, length=0.1, material=mat)

u1 = BlockWrapper(ChronoBody, width=0.1, length=0.1, material=mat)

# Transforms
RZX = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_X)
RZY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_Y)
RXY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_X_TO_Y)

MOVE_ZX_PLUS = ChCoordsysD(ChVectorD(0.3, 0, 0.3), ChQuaternionD(1, 0, 0, 0))
MOVE_ZX_MINUS = ChCoordsysD(ChVectorD(-0.3, 0, -0.3), ChQuaternionD(1, 0, 0, 0))

MOVE_X_PLUS = ChCoordsysD(ChVectorD(0.3, 0, 0), ChQuaternionD(1, 0, 0, 0))
MOVE_Z_PLUS_X_MINUS = ChCoordsysD(
    ChVectorD(-0.3, 0, 0.3), ChQuaternionD(1, 0, 0, 0))

transform_rzx = BlockWrapper(ChronoTransform, RZX)
transform_rzy = BlockWrapper(ChronoTransform, RZY)
transform_rxy = BlockWrapper(ChronoTransform, RXY)
transform_mzx_plus = BlockWrapper(ChronoTransform, MOVE_ZX_PLUS)
transform_mzx_minus = BlockWrapper(ChronoTransform, MOVE_ZX_MINUS)
transform_mx_plus = BlockWrapper(ChronoTransform, MOVE_X_PLUS)
transform_mz_plus_x_minus = BlockWrapper(ChronoTransform, MOVE_Z_PLUS_X_MINUS)

# Joints

type_of_input = ChronoRevolveJoint.InputType.Torque
revolve1 = BlockWrapper(
    ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)

# Nodes

J1 = Node(label="J1", is_terminal=True, block_wrapper=revolve1)
L1 = Node(label="L1", is_terminal=True, block_wrapper=link1)
L2 = Node(label="L2", is_terminal=True, block_wrapper=link2)
F1 = Node(label="F1", is_terminal=True, block_wrapper=flat1)
F2 = Node(label="F2", is_terminal=True, block_wrapper=flat2)
U1 = Node(label="U1", is_terminal=True, block_wrapper=u1)
T1 = Node(label="T1", is_terminal=True, block_wrapper=transform_mx_plus)
T2 = Node(label="T2", is_terminal=True, block_wrapper=transform_mz_plus_x_minus)
T3 = Node(label="T3", is_terminal=True, block_wrapper=transform_mzx_plus)
T4 = Node(label="T4", is_terminal=True, block_wrapper=transform_mzx_minus)


J = Node("J")
L = Node("L")
F = Node("F")
M = Node("M")
EF = Node("EF")
EM = Node("EM")

J_NODES = [J, J1]
B_NODES = [L, L1, L2, F1, F2]
T_EXAMPLE = [T1, T2]

# Defines rules

# Non terminal
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


rule_action_non_terminal_three_finger = np.asarray([FlatCreate, Mount, Mount, Mount,
                                       FingerUpper, FingerUpper, FingerUpper,
                                       FingerUpper,  FingerUpper, FingerUpper])
rule_action_terminal_three_finger = np.asarray([TerminalFlat,
                                   TerminalL1, TerminalL1, TerminalL1,
                                   TerminalL2, TerminalL2, TerminalL2,
                                   TerminalTransformL, TerminalTransformLZ,
                                   TerminalTransformRX,
                                   TerminalEndLimb, TerminalEndLimb,
                                   TerminalEndLimb,
                                   TerminalJoint, TerminalJoint, TerminalJoint,
                                   TerminalJoint, TerminalJoint, TerminalJoint])
rule_action_three_finger = np.r_[rule_action_non_terminal_three_finger, rule_action_terminal_three_finger]



rule_action_non_terminal_two_finger = np.asarray([FlatCreate, Mount, Mount,
                                       FingerUpper, FingerUpper, FingerUpper, FingerUpper,  FingerUpper])

rule_action_terminal_two_finger = np.asarray([TerminalFlat,
                         TerminalL1, TerminalL1, TerminalL1, TerminalL2, TerminalL2,
                         TerminalTransformL, TerminalTransformLZ,
                         TerminalEndLimb, TerminalEndLimb,
                         TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint])
rule_action_two_finger = np.r_[rule_action_non_terminal_two_finger, rule_action_terminal_two_finger]


rule_action_non_terminal_ladoshaka = np.asarray([FlatCreate,Mount])

rule_action_terminal_ladoshaka = np.asarray([TerminalFlat,TerminalTransformL,TerminalEndLimb])

rule_action_ladoshaka = np.r_[rule_action_non_terminal_ladoshaka, rule_action_terminal_ladoshaka]




def get_terminal_graph_three_finger():
    G = GraphGrammar()
    for i in list(rule_action_three_finger):
        G.apply_rule(i)
    return G

def get_terminal_graph_ladoshaka():
    G = GraphGrammar()
    for i in list(rule_action_ladoshaka):
        G.apply_rule(i)
    return G

def get_terminal_graph_two_finger():
    G = GraphGrammar()
    for i in list(rule_action_two_finger):
        G.apply_rule(i)
    return G