from time import sleep,time
import context

from engine.node  import BlockWrapper, Node, Rule, GraphGrammar, ROOT
from engine.node_render import *
from utils.flags_simualtions import FlagSlipout, FlagNotContact, FlagMaxTime
from utils.blocks_utils import make_collide, CollisionGroup   
from pychrono import ChCoordsysD, ChVectorD, ChQuaternionD
from pychrono import Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X, \
    Q_ROTATE_Y_TO_X, Q_ROTATE_Y_TO_Z, \
    Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z
import pychrono as chrono
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import simulation_step as step
import engine.robot as robot
import engine.control as control


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
MOVE_Z_PLUS_X_MINUS = ChCoordsysD(ChVectorD(-0.3, 0, 0.3), ChQuaternionD(1, 0, 0, 0))

transform_rzx = BlockWrapper(ChronoTransform, RZX)
transform_rzy = BlockWrapper(ChronoTransform, RZY)
transform_rxy = BlockWrapper(ChronoTransform, RXY)
transform_mzx_plus = BlockWrapper(ChronoTransform, MOVE_ZX_PLUS)
transform_mzx_minus = BlockWrapper(ChronoTransform, MOVE_ZX_MINUS)
transform_mx_plus = BlockWrapper(ChronoTransform, MOVE_X_PLUS)
transform_mz_plus_x_minus = BlockWrapper(ChronoTransform, MOVE_Z_PLUS_X_MINUS)

# Joints

type_of_input = ChronoRevolveJoint.InputType.Position
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
for i in list(rule_action):
    G.apply_rule(i)

chrono_system = chrono.ChSystemNSC()
grab_robot = robot.Robot(G, chrono_system)

joints = np.array(grab_robot.get_joints)
for m in range(6):
    if m == 0:
        traj_controller = np.array(np.mat('0 0.1 0.15 0.2 0.25 0.3; 0 0.2 0.4 0.6 0.8 1'))
    elif m != 1:
        traj_controller[0,:] *=2
        print(traj_controller)
        
    arr_traj = []
    for ind, finger in enumerate(joints):
        arr_finger_traj = []
        for i, joint in enumerate(finger):
            arr_finger_traj.append(traj_controller)
        arr_traj.append(arr_finger_traj)

    config_sys = {"Set_G_acc":chrono.ChVectorD(0,0,0)}

    time_to_contact = 2
    time_without_contact = 0.2
    max_time = 10


    flags = [FlagSlipout(time_to_contact,time_without_contact),
            FlagNotContact(time_to_contact), FlagMaxTime(max_time)]

    times_step = 1e-2


    obj = chrono.ChBodyEasyBox(0.2,0.2,0.6,1000,True,True,mat)
    obj.SetCollide(True)
    obj.SetPos(chrono.ChVectorD(0,1.2,0))
    
    sim = step.SimulationStepOptimization(arr_traj, G, obj)
    sim.set_flags_stop_simulation(flags)
    sim.change_config_system(config_sys)
    sim_output = sim.simulate_system(times_step,True)
    print(m)
    print(sim_output[40].sum_contact_forces)