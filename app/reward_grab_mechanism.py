from time import time

from rostok.graph_grammar.node import BlockWrapper, Node, Rule, GraphGrammar, ROOT
from rostok.block_builder.node_render import *
from rostok.criterion.flags_simualtions import FlagSlipout, FlagNotContact, FlagMaxTime
from pychrono import ChCoordsysD, ChVectorD
from pychrono import Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X, \
    Q_ROTATE_X_TO_Y
from rostok.graph_grammar.nodes_division import *
from rostok.criterion.criterion_calc import *
import pychrono as chrono
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import rostok.virtual_experiment.simulation_step as simulation_step
import rostok.virtual_experiment.robot as robot
from numpy import arange
from rostok.block_builder.blocks_utils import NodeFeatures


def is_body(node: Node):
    return node in list_B


def plot_graph(graph):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(G, dim=2),
                     node_size=800,
                     labels={n: G.nodes[n]["Node"].label for n in G})
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axis([0, 10, 0, 10])
    ax.text(2,
            8,
            'Close all matplotlib for start simlation',
            style='italic',
            fontsize=15,
            bbox={
                'facecolor': 'red',
                'alpha': 0.5,
                'pad': 10
            })

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
link1 = BlockWrapper(BasicChronoBody, length=0.5, mass=0.3)
link2 = BlockWrapper(BasicChronoBody, length=0.5, mass=0.3)

flat1 = BlockWrapper(FlatChronoBody, width=0.5, length=0.1, depth=0.8)
flat2 = BlockWrapper(FlatChronoBody, width=0.5, length=0.1, depth=0.8)

u1 = BlockWrapper(MountChronoBody, length=0.05, mass=0.01)

# Transforms
RZX = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_X)
RZY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_Y)
RXY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_X_TO_Y)

MOVE_ZX_PLUS = FrameTransform([0.3, 0, 0.3], [1, 0, 0, 0])
MOVE_ZX_MINUS = FrameTransform([-0.3, 0, -0.3], [1, 0, 0, 0])

MOVE_X_PLUS = FrameTransform([0.3, 0, 0.], [1, 0, 0, 0])
MOVE_Z_PLUS_X_MINUS = FrameTransform([-0.3, 0, 0.3], [1, 0, 0, 0])

transform_rzx = BlockWrapper(ChronoTransform, RZX)
transform_rzy = BlockWrapper(ChronoTransform, RZY)
transform_rxy = BlockWrapper(ChronoTransform, RXY)
transform_mzx_plus = BlockWrapper(ChronoTransform, MOVE_ZX_PLUS)
transform_mzx_minus = BlockWrapper(ChronoTransform, MOVE_ZX_MINUS)
transform_mx_plus = BlockWrapper(ChronoTransform, MOVE_X_PLUS)
transform_mz_plus_x_minus = BlockWrapper(ChronoTransform, MOVE_Z_PLUS_X_MINUS)

# Joints

type_of_input = ChronoRevolveJoint.InputType.Torque
revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z, type_of_input)

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
list_Palm = [F1]

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

rule_action_non_terminal = np.asarray([
    FlatCreate, Mount, Mount, Mount, FingerUpper, FingerUpper, FingerUpper, FingerUpper,
    FingerUpper, FingerUpper
])
rule_action_terminal = np.asarray([
    TerminalFlat, TerminalL1, TerminalL1, TerminalL1, TerminalL2, TerminalL2, TerminalL2,
    TerminalTransformL, TerminalTransformLZ, TerminalTransformRX, TerminalEndLimb, TerminalEndLimb,
    TerminalEndLimb, TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint, TerminalJoint,
    TerminalJoint
])
rule_action = np.r_[rule_action_non_terminal, rule_action_terminal]

for i in list(rule_action):
    G.apply_rule(i)

#Set type of system

chrono_system = chrono.ChSystemNSC()
# chrono_system = chrono.ChSystemSMC()

grab_robot = robot.Robot(G, chrono_system)

obj = BlockWrapper(ChronoBodyEnv, width=0.3, depth=0.6, length=0.3, pos = FrameTransform([0, 0.8, 0], [1, 0, 0, 0]))

node_list_plain = list(map(G.get_node_by_id, G.get_ids_in_dfs_order()))

config_sys = {"Set_G_acc": chrono.ChVectorD(0, -10, 0)}


def create_const_traj(torque_value, stop_time: float, time_step: float):
    timeseries_traj = []
    timeseries = list(arange(0, stop_time, time_step))
    traj = [torque_value for _ in timeseries]
    timeseries_traj.append(timeseries)
    timeseries_traj.append(traj)
    return timeseries_traj


def create_torque_traj_from_x(joint_dfs, x: list[float], stop_time: float, time_step: float):
    x_iter = iter(x)
    torque_traj = []
    for branch in joint_dfs:
        control_one_branch = []
        for block in branch:
            one_torque = next(x_iter)
            control_one_branch.append(create_const_traj(one_torque, stop_time, time_step))
        torque_traj.append(np.array(control_one_branch))

    return torque_traj


dfs_patrion_ids = G.graph_partition_dfs()


def get_node(node_id):
    return G.get_node_by_id(node_id)


dfs_patrion_node = [[get_node(node_id) for node_id in branch] for branch in dfs_patrion_ids]
dfs_j = []
number_trq = 0
for branch in dfs_patrion_node:
    joint_branch = list(filter(NodeFeatures.is_joint, branch))
    len_joints = len(joint_branch)
    number_trq += len_joints
    if len_joints != 0:
        dfs_j.append(joint_branch)

const_torque_koef = [random.random() for _ in range(number_trq)]
arr_trj = create_torque_traj_from_x(dfs_j, const_torque_koef, 10, 0.1)

time_model = time()
time_to_contact = 2
time_without_contact = 0.2
max_time = 10
flags = [
    FlagSlipout(time_to_contact, time_without_contact),
    FlagNotContact(time_to_contact),
    FlagMaxTime(max_time)
]

times_step = 1e-3

WEIGHTS = [5, 1, 1, 5]
GAIT_PERIOD = 2.5

# if __name__ == '__main__':
sim = simulation_step.SimulationStepOptimization(arr_trj, G, obj)
J_NODES_NEW = nodes_division(sim.grab_robot, list_J)
B_NODES_NEW = nodes_division(sim.grab_robot, list_B)
RB_NODES_NEW = sort_left_right(sim.grab_robot, list_RM, list_B)
LB_NODES_NEW = sort_left_right(sim.grab_robot, list_LM, list_B)
RJ_NODES_NEW = sort_left_right(sim.grab_robot, list_RM, list_J)
LJ_NODES_NEW = sort_left_right(sim.grab_robot, list_LM, list_J)
sim.set_flags_stop_simulation(flags)
sim.change_config_system(config_sys)
sim_output = sim.simulate_system(times_step, True)

reward = criterion_calc(sim_output, B_NODES_NEW, J_NODES_NEW, LB_NODES_NEW, RB_NODES_NEW, WEIGHTS,
                        GAIT_PERIOD)
print(reward)
