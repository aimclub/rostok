from ast import Break
from time import sleep
import context

from engine.node  import BlockWrapper, Node, Rule, GraphGrammar, ROOT
from engine import node_vocabulary, rule_vocabulary
from utils.blocks_utils import make_collide, CollisionGroup   
from engine.node_render import *
from nodes_division import *
from sort_left_right import *
# from find_traj_fun import *
import engine.robot as robot
import engine.control as ctrl


from pychrono import ChCoordsysD, ChVectorD, ChQuaternionD
from pychrono import Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X, \
    Q_ROTATE_Y_TO_X, Q_ROTATE_Y_TO_Z, \
    Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z
import pychrono as chrono

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_graph(graph):
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800,
                    labels={n: G.nodes[n]["Node"].label for n in G})
    plt.figure()
    #nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800)
    #plt.show()
    

# Define materials
mat = chrono.ChMaterialSurfaceNSC()
mysystem = chrono.ChSystemNSC() 
mat.SetFriction(0.5)


# %% Bodies for extansions rules
width = [0.5, 0.75, 1.]
alpha = 45
length_link = [0.5, 0.8, 1.]

flat = list(map(lambda x: BlockWrapper(ChronoBody, width=x, length=0.01),
                width))

link = list(map(lambda x: BlockWrapper(ChronoBody, length=x),
                length_link))


u1 = BlockWrapper(ChronoBody, width=0.1, length=0.0)


# %% Transforms
RZX = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_X)
RZY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_Y)
RXY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_X_TO_Y)


# %% Tranform for extansions rules 

# MOVE_TO_RIGHT_SIDE = map(lambda x: ChCoordsysD(ChVectorD(x, 0, 0), chrono.Q_FLIP_AROUND_Y),
#                 width)
# MOVE_TO_LEFT_SIDE = map(lambda x: ChCoordsysD(ChVectorD(-x, 0, 0), chrono.QUNIT),
#                 width)

# ROTATE_TO_ALPHA = ChCoordsysD(ChVectorD(0, 0, 0), chrono.Q_from_AngY(np.deg2rad(alpha)))


MOVE_TO_RIGHT_SIDE = map(lambda x: {"pos":[x, 0, 0],"rot":[0,0,1,0]},
                width)
MOVE_TO_RIGHT_SIDE_PLUS = map(lambda x: {"pos":[x, 0, +0.5],"rot":[0,0,1,0]},
                width)
MOVE_TO_RIGHT_SIDE_MINUS = map(lambda x: {"pos":[x, 0, -0.5],"rot":[0,0,1,0]},
                width)
MOVE_TO_LEFT_SIDE = map(lambda x:  {"pos":[-x, 0, 0],"rot":[1,0,0,0]},
                width)
MOVE_TO_LEFT_SIDE_PLUS = map(lambda x: {"pos":[x, 0, +0.5],"rot":[0,0,1,0]},
                width)
MOVE_TO_LEFT_SIDE_MINUS = map(lambda x: {"pos":[x, 0, -0.5],"rot":[0,0,1,0]},
                width)

quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
ROTATE_TO_ALPHA = {"pos":[0, 0, 0],"rot":[quat_Y_ang_alpha.e0,quat_Y_ang_alpha.e1,
                                          quat_Y_ang_alpha.e2,quat_Y_ang_alpha.e3]}


transform_to_right_mount = list(map(lambda x: BlockWrapper(ChronoTransform, x),
                MOVE_TO_RIGHT_SIDE))
transform_to_right_mount_plus = list(map(lambda x: BlockWrapper(ChronoTransform, x),
                MOVE_TO_RIGHT_SIDE_PLUS))
transform_to_right_mount_minus = list(map(lambda x: BlockWrapper(ChronoTransform, x),
                MOVE_TO_RIGHT_SIDE_MINUS))
transform_to_left_mount = list(map(lambda x: BlockWrapper(ChronoTransform, x),
                MOVE_TO_LEFT_SIDE))
transform_to_left_mount_plus = list(map(lambda x: BlockWrapper(ChronoTransform, x),
                MOVE_TO_LEFT_SIDE_PLUS))
transform_to_left_mount_minus = list(map(lambda x: BlockWrapper(ChronoTransform, x),
                MOVE_TO_LEFT_SIDE_MINUS))
transform_to_alpha_rotate = BlockWrapper(ChronoTransform, ROTATE_TO_ALPHA)


# %%
type_of_input = ChronoRevolveJoint.InputType.Position

# Joints
revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)

# Nodes
node_vocab = node_vocabulary.NodeVocabulary()
node_vocab.add_node(ROOT)
node_vocab.create_node("J")
node_vocab.create_node("L")
node_vocab.create_node("F")
node_vocab.create_node("M")
node_vocab.create_node("EF")
node_vocab.create_node("EM")
node_vocab.create_node("SML")
node_vocab.create_node("SMR")
node_vocab.create_node("SMR1")
node_vocab.create_node("SML1")
node_vocab.create_node("SMR2")
node_vocab.create_node("SML2")


#O = Node("O")
node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=revolve1)
node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=link[0])
node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=link[1])
node_vocab.create_node(label="L3", is_terminal=True, block_wrapper=link[2])
node_vocab.create_node(label="F1", is_terminal=True, block_wrapper=flat[0])
node_vocab.create_node(label="F2", is_terminal=True, block_wrapper=flat[1])
node_vocab.create_node(label="F3", is_terminal=True, block_wrapper=flat[2])
node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=u1)

node_vocab.create_node(label="TR1", is_terminal=True, block_wrapper=transform_to_right_mount[0])
node_vocab.create_node(label="TR2", is_terminal=True, block_wrapper=transform_to_right_mount[1])
node_vocab.create_node(label="TR3", is_terminal=True, block_wrapper=transform_to_right_mount[2])
node_vocab.create_node(label="TR1_plus", is_terminal=True, block_wrapper=transform_to_right_mount_plus[0])
node_vocab.create_node(label="TR2_plus", is_terminal=True, block_wrapper=transform_to_right_mount_plus[1])
node_vocab.create_node(label="TR3_plus", is_terminal=True, block_wrapper=transform_to_right_mount_plus[2])
node_vocab.create_node(label="TR1_minus", is_terminal=True, block_wrapper=transform_to_right_mount_minus[0])
node_vocab.create_node(label="TR2_minus", is_terminal=True, block_wrapper=transform_to_right_mount_minus[1])
node_vocab.create_node(label="TR3_minus", is_terminal=True, block_wrapper=transform_to_right_mount_minus[2])

node_vocab.create_node(label="TL1", is_terminal=True, block_wrapper=transform_to_left_mount[0])
node_vocab.create_node(label="TL2", is_terminal=True, block_wrapper=transform_to_left_mount[1])
node_vocab.create_node(label="TL3", is_terminal=True, block_wrapper=transform_to_left_mount[2])
node_vocab.create_node(label="TL1_plus", is_terminal=True, block_wrapper=transform_to_left_mount_plus[0])
node_vocab.create_node(label="TL2_plus", is_terminal=True, block_wrapper=transform_to_left_mount_plus[1])
node_vocab.create_node(label="TL3_plus", is_terminal=True, block_wrapper=transform_to_left_mount_plus[2])
node_vocab.create_node(label="TL1_minus", is_terminal=True, block_wrapper=transform_to_left_mount_minus[0])
node_vocab.create_node(label="TL2_minus", is_terminal=True, block_wrapper=transform_to_left_mount_minus[1])
node_vocab.create_node(label="TL3_minus", is_terminal=True, block_wrapper=transform_to_left_mount_minus[2])


rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

rule_vocab.create_rule("InitMechanism_2", ["ROOT"], ["F", "SML", "SMR","EM","EM"], 0 , 0)


list_J = [J1]
list_RM = [TR1, TR2, TR3]
list_LM = [TL1, TL2, TL3]
list_B = [L1, L2, L3, F1, F2, F3]



# Defines rules

# %%
# Non terminal
# Simple replace

InitMechanism = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=F)
rule_graph.add_node(1, Node=SML)
rule_graph.add_node(2, Node=SMR)
rule_graph.add_node(3, Node=EM)
rule_graph.add_node(4, Node=EM)
rule_graph.add_edge(0, 1)
rule_graph.add_edge(0, 2)
rule_graph.add_edge(1, 3)
rule_graph.add_edge(2, 4)
InitMechanism.id_node_connect_child = 0
InitMechanism.id_node_connect_parent = 6
InitMechanism.graph_insert = rule_graph
InitMechanism.replaced_node = ROOT

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

# FingerUpper_N = Rule()
# rule_graph = nx.DiGraph()
# rule_graph.add_node(0, Node=O)
# rule_graph.add_node(1, Node=J)
# rule_graph.add_node(2, Node=L)
# rule_graph.add_node(3, Node=EM)
# rule_graph.add_edge(0, 1)
# rule_graph.add_edge(1, 2)
# rule_graph.add_edge(2, 3)
# FingerUpper_N.id_node_connect_child = 3
# FingerUpper_N.id_node_connect_parent = 0
# FingerUpper_N.graph_insert = rule_graph
# FingerUpper_N.replaced_node = EM


# %% Terminal rules
TerminalFlat1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=F1)
TerminalFlat1.id_node_connect_child = 0
TerminalFlat1.id_node_connect_parent = 0
TerminalFlat1.graph_insert = rule_graph
TerminalFlat1.replaced_node = F

TerminalFlat2 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=F2)
TerminalFlat2.id_node_connect_child = 0
TerminalFlat2.id_node_connect_parent = 0
TerminalFlat2.graph_insert = rule_graph
TerminalFlat2.replaced_node = F

TerminalFlat3 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=F3)
TerminalFlat3.id_node_connect_child = 0
TerminalFlat3.id_node_connect_parent = 0
TerminalFlat3.graph_insert = rule_graph
TerminalFlat3.replaced_node = F

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

TerminalL3 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=L3)
TerminalL3.id_node_connect_child = 0
TerminalL3.id_node_connect_parent = 0
TerminalL3.graph_insert = rule_graph
TerminalL3.replaced_node = L

TerminalTransformRight1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=TR1)
TerminalTransformRight1.id_node_connect_child = 0
TerminalTransformRight1.id_node_connect_parent = 0
TerminalTransformRight1.graph_insert = rule_graph
TerminalTransformRight1.replaced_node = SMR

TerminalTransformRight2 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=TR2)
TerminalTransformRight2.id_node_connect_child = 0
TerminalTransformRight2.id_node_connect_parent = 0
TerminalTransformRight2.graph_insert = rule_graph
TerminalTransformRight2.replaced_node = SMR

TerminalTransformRight3 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=TR3)
TerminalTransformRight3.id_node_connect_child = 0
TerminalTransformRight3.id_node_connect_parent = 0
TerminalTransformRight3.graph_insert = rule_graph
TerminalTransformRight3.replaced_node = SMR

TerminalTransformLeft1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=TL1)
TerminalTransformLeft1.id_node_connect_child = 0
TerminalTransformLeft1.id_node_connect_parent = 0
TerminalTransformLeft1.graph_insert = rule_graph
TerminalTransformLeft1.replaced_node = SML

TerminalTransformLeft2 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=TL2)
TerminalTransformLeft2.id_node_connect_child = 0
TerminalTransformLeft2.id_node_connect_parent = 0
TerminalTransformLeft2.graph_insert = rule_graph
TerminalTransformLeft2.replaced_node = SML

TerminalTransformLeft3 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=TL3)
TerminalTransformLeft3.id_node_connect_child = 0
TerminalTransformLeft3.id_node_connect_parent = 0
TerminalTransformLeft3.graph_insert = rule_graph
TerminalTransformLeft3.replaced_node = SML


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

rule_action_non_terminal = np.asarray([InitMechanism,
                                        FingerUpper, FingerUpper,
                                        FingerUpper, FingerUpper,
                                        FingerUpper, FingerUpper,
                                        FingerUpper, FingerUpper])
rule_action_terminal = np.asarray([TerminalFlat1,
                                    TerminalJoint, TerminalJoint,
                                    TerminalL3, TerminalL3, 
                                    TerminalTransformRight3, TerminalTransformLeft3, 
                                    TerminalJoint, TerminalJoint,
                                    TerminalL2, TerminalL2,
                                    TerminalJoint, TerminalJoint,
                                    TerminalL1, TerminalL1,
                                    TerminalJoint, TerminalJoint,
                                    TerminalL1, TerminalL1,
                                    TerminalEndLimb, TerminalEndLimb])
rule_action = np.r_[rule_action_non_terminal, rule_action_terminal]


for i in list(rule_action):
    G.apply_rule(i)

plot_graph(G)
# %%
mysystem.Set_G_acc(chrono.ChVectorD(0,-1,0))

# Set solver settings
mysystem.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
mysystem.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
mysystem.SetMaxPenetrationRecoverySpeed(0.1)
mysystem.SetSolverMaxIterations(100)
mysystem.SetSolverForceTolerance(0)


robot = robot.Robot(G, mysystem)
joint_blocks = robot.get_joints

obj = chrono.ChBodyEasyCylinder(0.5,0.5,1000,True,True,mat)
# obj = chrono.ChBodyEasySphere(0.5,1000,True,True,mat)
obj.SetCollide(True)
obj.SetMass(2)
obj.SetRot(chrono.ChQuaternionD(0.707, 0.707, 0, 0))
obj.SetPos(chrono.ChVectorD(0,0.55,0))

mysystem.Add(obj)

base_id = robot.get_block_graph().find_nodes(F1)[0]
robot.block_map[base_id].body.SetBodyFixed(True)

#
# Make robot collide
blocks = robot.block_map.values()
body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
make_collide(body_block, CollisionGroup.Robot)
list_block = list(body_block)
force_list = []



# %%
T_0 = 1
T_1 = T_0

joint_blocks = robot.get_joints
const_trq = []
const_trq.append(ctrl.ConstControl(joint_blocks[0][0], T_1)) 
const_trq.append(ctrl.ConstControl(joint_blocks[0][1], T_1*0.7)) 
const_trq.append(ctrl.ConstControl(joint_blocks[0][2], T_1*0.5*0.7)) 
const_trq.append(ctrl.ConstControl(joint_blocks[0][3], T_1*0.5*0.5*0.7)) 
const_trq.append(ctrl.ConstControl(joint_blocks[1][0], T_0))
const_trq.append(ctrl.ConstControl(joint_blocks[1][1], T_0*0.7)) 
const_trq.append(ctrl.ConstControl(joint_blocks[1][2], T_0*0.5*0.7)) 
const_trq.append(ctrl.ConstControl(joint_blocks[1][3], T_0*0.5*0.5*0.7)) 


# link_L2_id = robot.get_block_graph().find_nodes(L2)[0]


# plt.figure()
# nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800,
#                  labels={n: G.nodes[n]["Node"].label for n in G})
# # plt.figure()
# # nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800)
# plt.show()

vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(mysystem)
vis.SetWindowSize(1024,768)
vis.SetWindowTitle('Custom contact demo')
vis.Initialize()
vis.AddCamera(chrono.ChVectorD(1, 1, -2))
vis.AddTypicalLights()

# %%


ext_force = chrono.ChForce()
ext_force.SetF_y(chrono.ChFunction_Const(np.random.randint(-3, 3)))
ext_force.SetF_x(chrono.ChFunction_Const(np.random.randint(-3, 3)))
ext_force.SetF_z(chrono.ChFunction_Const(np.random.randint(-3, 3)))
# obj.AddForce(ext_force)

partition_dfs = robot.get_dfs_partiton()

J_NODES_NEW = nodes_division(robot, list_J)
B_NODES_NEW = nodes_division(robot, list_B)
LM_NODES_NEW = nodes_division(robot, list_LM)
RM_NODES_NEW = nodes_division(robot, list_RM)
abcd = []
# for i in range(len(J_NODES_NEW)):
#     abcd.append(J_NODES_NEW[i].id)
abcd.append(filter(lambda x: x.id == 23, J_NODES_NEW))
    


LB_NODES_NEW = sort_left_right(robot, list_LM, list_B)
RB_NODES_NEW = sort_left_right(robot, list_RM, list_B)

test = [[23] ,[0.1, 0.2, 0.3, 0.4]]

# abcd = find_traj_fun(J_NODES, test)

cont = []
f_sum = []
ovr = []

while vis.Run():
    
    mysystem.Update()
    for i in range(10):
        mysystem.DoStepDynamics(1e-4)
    vis.BeginScene(True, True, chrono.ChColor(0.3, 0.2, 0.3))
    vis.Render()
    cont.clear()
    vis.EndScene()