from time import sleep,time
import context

from engine.node  import BlockWrapper, Node, Rule, GraphGrammar, ROOT
from engine.node_render import *
from utils.flags_simualtions import FlagSlipout, FlagNotContact, FlagMaxTime
from utils.blocks_utils import make_collide, CollisionGroup
from utils.auxilarity_sensors import RobotSensor   
from pychrono import ChCoordsysD, ChVectorD, ChQuaternionD
from pychrono import Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X, \
    Q_ROTATE_Y_TO_X, Q_ROTATE_Y_TO_Z, \
    Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z
from utils.nodes_division import *
from utils.criterion_calc import *
from scipy.spatial import distance
import pychrono as chrono
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import simulation_step as step
import engine.robot as robot
import engine.control as control


def is_body(node: Node): return node in list_B


def plot_graph(graph):
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800,
                    labels={n: G.nodes[n]["Node"].label for n in G})
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axis([0, 10, 0, 10])
    ax.text(2, 8, 'Close all matplotlib for start simlation', style='italic', fontsize=15,
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    plt.show()


# Define block types
# mat = chrono.ChMaterialSurfaceSMC()
# mat.SetFriction(0.4)
# mat.SetYoungModulus(100e5)
# mat.SetPoissonRatio(0.33)
# mat.SetKn(10)
# mat.SetGn(0.1)
# mat.SetRollingFriction(1e-3)
# mat.SetSpinningFriction(1e-3)
# mat.SetRestitution(0.01) #Coef. of restitution in range [0;1] where 1 is perfectly elastic collision; 0 is inelastic collision

mat = chrono.ChMaterialSurfaceNSC()
mat.SetFriction(0.5)
mat.SetDampingF(0.1)


# Bodies
link1 = BlockWrapper(ChronoBody, length=0.5, material=mat)
link2 = BlockWrapper(ChronoBody, length=0.5, material=mat)

flat1 = BlockWrapper(ChronoBody, width=0.4, length=0.1, material=mat)
flat2 = BlockWrapper(ChronoBody, width=0.7, length=0.1, material=mat)

u1 = BlockWrapper(ChronoBody, width=0.1, length=0.1, material=mat)

# Transforms
RZX = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_X)
RZY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_Y)
RXY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_X_TO_Y)

MOVE_ZX_PLUS = ChCoordsysD(ChVectorD(0.3, 0, 0.3), ChQuaternionD(1, 0, 0, 0))
MOVE_ZX_MINUS = ChCoordsysD(ChVectorD(-0.3, 0, -0.3), ChQuaternionD(1, 0, 0, 0))

MOVE_X_PLUS = ChCoordsysD(ChVectorD(0.3, 0, 0.3), ChQuaternionD(1, 0, 0, 0))
MOVE_Z_PLUS_X_MINUS = ChCoordsysD(ChVectorD(-0.3, 0, 0.3), ChQuaternionD(1, 0, 0, 0))

transform_rzx = BlockWrapper(ChronoTransform, RZX)
transform_rzy = BlockWrapper(ChronoTransform, RZY)
transform_rxy = BlockWrapper(ChronoTransform, RXY)
transform_mzx_plus = BlockWrapper(ChronoTransform, MOVE_ZX_PLUS)
transform_mzx_minus = BlockWrapper(ChronoTransform, MOVE_ZX_MINUS)
transform_mx_plus = BlockWrapper(ChronoTransform, MOVE_X_PLUS)
transform_mz_plus_x_minus = BlockWrapper(ChronoTransform, MOVE_Z_PLUS_X_MINUS)

# Joints

type_of_input = ChronoRevolveJoint.InputType.Torque
revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)

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

#Lists of possible nodes
list_J = [J1]
list_RM = [T1, T3]
list_LM = [T2, T4]
list_B = [L1, L2, F1, F2, U1]

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


G = GraphGrammar()

rule_action_non_terminal = np.asarray([FlatCreate, Mount, Mount, Mount,
                                       FingerUpper, FingerUpper, FingerUpper, FingerUpper,  FingerUpper, FingerUpper])
rule_action_terminal = np.asarray([TerminalFlat,
                         TerminalL1, TerminalL1, TerminalL1, TerminalL2, TerminalL2, TerminalL2,
                         TerminalTransformL, TerminalTransformLZ, TerminalTransformRX,
                         TerminalEndLimb, TerminalEndLimb, TerminalEndLimb,
                         TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint])
rule_action = np.r_[rule_action_non_terminal, rule_action_terminal]


# Test for 2 fingers:

# rule_action_non_terminal = np.asarray([FlatCreate, Mount, Mount,
#                                        FingerUpper, FingerUpper, FingerUpper, FingerUpper])
# rule_action_terminal = np.asarray([TerminalFlat,
#                          TerminalL1, TerminalL1, TerminalL2, TerminalL2,
#                          TerminalTransformL, TerminalTransformRX,
#                          TerminalEndLimb, TerminalEndLimb,
#                          TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint])
# rule_action = np.r_[rule_action_non_terminal, rule_action_terminal]


for i in list(rule_action):
    G.apply_rule(i)


#Set type of system 

# chrono_system = chrono.ChSystemNSC()
chrono_system = chrono.ChSystemSMC()

grab_robot = robot.Robot(G, chrono_system)


joints = np.array(grab_robot.get_joints)
for m in range(6):
    if m == 0:
        traj_controller = np.array(np.mat('0 0.3 0.6 0.9 1.2 2; 0.5 0.5 0.5 0.5 0.5 0.5')) #Format: [Time; Value].
        # traj_controller = np.array(np.mat('0 0.3 0.6 0.9 1.2 2; 0.0 0.0 0.0 0.0 0.0 0.0 ')) #Format: [Time; Value].
    elif m != 1:
        traj_controller[1,:] *=2
        print(traj_controller)
        
    arr_traj = []
    for ind, finger in enumerate(joints):
        arr_finger_traj = []
        for i, joint in enumerate(finger):
            arr_finger_traj.append(traj_controller)
        arr_traj.append(arr_finger_traj)

obj = chrono.ChBodyEasyBox(0.2,0.2,0.6,1000,True,True,mat)
obj.SetCollide(True)
obj.SetPos(chrono.ChVectorD(0,0.5,0))
obj.SetBodyFixed(True)

# plt.figure()
# nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800,
#                  labels={n: G.nodes[n]["Node"].label for n in G})
# plt.figure()
# nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800)
# plt.show()

node_list_plain = list(map(G.get_node_by_id,
                      G.get_ids_in_dfs_order()))

config_sys = {"Set_G_acc":chrono.ChVectorD(0,-10,0)}

#Create sorted lists 
J_NODES_NEW = nodes_division(grab_robot, list_J)
B_NODES_NEW = nodes_division(grab_robot, list_B)
RB_NODES_NEW = sort_left_right(grab_robot, list_RM, list_B)
LB_NODES_NEW = sort_left_right(grab_robot, list_LM, list_B)
RJ_NODES_NEW = sort_left_right(grab_robot, list_RM, list_J)
LJ_NODES_NEW = sort_left_right(grab_robot, list_LM, list_J)

RB_blocks = [B_NODES_NEW[0].block]
LB_blocks = [B_NODES_NEW[0].block]
for i in range(len(RB_NODES_NEW)):
    for j in range(len(RB_NODES_NEW[i])):
        RB_blocks.append(RB_NODES_NEW[i][j].block)

for i in range(len(LB_NODES_NEW)):
    for j in range(len(LB_NODES_NEW[i])):
        LB_blocks.append(LB_NODES_NEW[i][j].block)

# test_block = []
# for i in range(len(RB_NODES_NEW)):
#     test_block.append(list(filter(is_body,RB_NODES_NEW[i])))

time_model = time()

time_to_contact = 2
time_without_contact = 0.2
max_time = 10
flags = [FlagSlipout(time_to_contact,time_without_contact),
         FlagNotContact(time_to_contact), FlagMaxTime(max_time)]

times_step = 1e-3


sim = step.SimulationStepOptimization(arr_traj, G, obj, RB_blocks, LB_blocks)
sim.set_flags_stop_simulation(flags)
sim.change_config_system(config_sys)
sim_output = sim.simulate_system(times_step, True)


[B_NODES_NEW, J_NODES_NEW, LB_NODES_NEW, RB_NODES_NEW]  = traj_to_list(B_NODES_NEW, J_NODES_NEW, LB_NODES_NEW, RB_NODES_NEW, sim_output)
reward = criterion_calc(B_NODES_NEW, J_NODES_NEW, LB_NODES_NEW, RB_NODES_NEW)
print(reward)
print(None)