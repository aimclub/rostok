from ast import Break
from time import sleep
import context
import mcts

from engine.node  import BlockWrapper, Node, Rule, GraphGrammar, ROOT
from engine import node_vocabulary, rule_vocabulary
from utils.blocks_utils import make_collide, CollisionGroup   
from engine.node_render import *
from stubs.graph_reward import Reward
import stubs.graph_environment as env_graph
#from nodes_division import *
#from sort_left_right import *
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
    #plt.figure()
    #nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800)
    plt.show()
    

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
MOVE_TO_LEFT_SIDE_PLUS = map(lambda x: {"pos":[-x, 0, +0.5],"rot":[1,0,0,0]},
                width)
MOVE_TO_LEFT_SIDE_MINUS = map(lambda x: {"pos":[-x, 0, -0.5],"rot":[1,0,0,0]},
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
node_vocab.create_node("SMRP")
node_vocab.create_node("SMLP")
node_vocab.create_node("SMRM")
node_vocab.create_node("SMLM")


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
node_vocab.create_node(label="TRP1", is_terminal=True, block_wrapper=transform_to_right_mount_plus[0])
node_vocab.create_node(label="TRP2", is_terminal=True, block_wrapper=transform_to_right_mount_plus[1])
node_vocab.create_node(label="TRP3", is_terminal=True, block_wrapper=transform_to_right_mount_plus[2])
node_vocab.create_node(label="TRM1", is_terminal=True, block_wrapper=transform_to_right_mount_minus[0])
node_vocab.create_node(label="TRM2", is_terminal=True, block_wrapper=transform_to_right_mount_minus[1])
node_vocab.create_node(label="TRM3", is_terminal=True, block_wrapper=transform_to_right_mount_minus[2])

node_vocab.create_node(label="TL1", is_terminal=True, block_wrapper=transform_to_left_mount[0])
node_vocab.create_node(label="TL2", is_terminal=True, block_wrapper=transform_to_left_mount[1])
node_vocab.create_node(label="TL3", is_terminal=True, block_wrapper=transform_to_left_mount[2])
node_vocab.create_node(label="TLP1", is_terminal=True, block_wrapper=transform_to_left_mount_plus[0])
node_vocab.create_node(label="TLP2", is_terminal=True, block_wrapper=transform_to_left_mount_plus[1])
node_vocab.create_node(label="TLP3", is_terminal=True, block_wrapper=transform_to_left_mount_plus[2])
node_vocab.create_node(label="TLM1", is_terminal=True, block_wrapper=transform_to_left_mount_minus[0])
node_vocab.create_node(label="TLM2", is_terminal=True, block_wrapper=transform_to_left_mount_minus[1])
node_vocab.create_node(label="TLM3", is_terminal=True, block_wrapper=transform_to_left_mount_minus[2])

# Defines rules
rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

rule_vocab.create_rule("InitMechanism_2", ["ROOT"], ["F", "SML", "SMR","EM","EM"], 0 , 0,[(0,1),(0,2),(1,3),(2,4)])
rule_vocab.create_rule("InitMechanism_3_R", ["ROOT"], ["F", "SML", "SMRP","SMRM","EM","EM","EM"], 0 , 0,[(0,1),(0,2),(0,3),(1,4),(2,5),(3,6)])
rule_vocab.create_rule("InitMechanism_3_L", ["ROOT"], ["F", "SMLP","SMLM", "SMR","EM","EM","EM"], 0 , 0, [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6)])
rule_vocab.create_rule("InitMechanism_4", ["ROOT"], ["F", "SMLP","SMLM", "SMRP","SMRM","EM","EM","EM","EM"], 0 , 0, [(0,1),(0,2),(0,3),(0,4),(1,5),(2,6),(3,7),(4,8)])
rule_vocab.create_rule("FingerUpper", ["EM"], ["J", "L","EM"], 0 , 2, [(0,1),(1, 2)])

rule_vocab.create_rule("TerminalFlat1", ["F"], ["F1"], 0 , 0)
rule_vocab.create_rule("TerminalFlat2", ["F"], ["F2"], 0 , 0)
rule_vocab.create_rule("TerminalFlat3", ["F"], ["F3"], 0 , 0)

rule_vocab.create_rule("TerminalL1", ["L"], ["L1"], 0 , 0)
rule_vocab.create_rule("TerminalL2", ["L"], ["L2"], 0 , 0)
rule_vocab.create_rule("TerminalL3", ["L"], ["L3"], 0 , 0)

rule_vocab.create_rule("TerminalTransformRight1", ["SMR"], ["TR1"], 0 , 0)
rule_vocab.create_rule("TerminalTransformRight2", ["SMR"], ["TR2"], 0 , 0)
rule_vocab.create_rule("TerminalTransformRight3", ["SMR"], ["TR3"], 0 , 0)

rule_vocab.create_rule("TerminalTransformRightPlus1", ["SMRP"], ["TRP1"], 0 , 0)
rule_vocab.create_rule("TerminalTransformRightPlus2", ["SMRP"], ["TRP2"], 0 , 0)
rule_vocab.create_rule("TerminalTransformRightPlus3", ["SMRP"], ["TRP3"], 0 , 0)

rule_vocab.create_rule("TerminalTransformRightMinus1", ["SMRM"], ["TRM1"], 0 , 0)
rule_vocab.create_rule("TerminalTransformRightMinus2", ["SMRM"], ["TRM2"], 0 , 0)
rule_vocab.create_rule("TerminalTransformRightMinus3", ["SMRM"], ["TRM3"], 0 , 0)

rule_vocab.create_rule("TerminalTransformLeft1", ["SML"], ["TL1"], 0 , 0)
rule_vocab.create_rule("TerminalTransformLeft2", ["SML"], ["TL2"], 0 , 0)
rule_vocab.create_rule("TerminalTransformLeft3", ["SML"], ["TL3"], 0 , 0)

rule_vocab.create_rule("TerminalTransformLeftPlus1", ["SMLP"], ["TLP1"], 0 , 0)
rule_vocab.create_rule("TerminalTransformLeftPlus2", ["SMLP"], ["TLP2"], 0 , 0)
rule_vocab.create_rule("TerminalTransformLeftPlus3", ["SMLP"], ["TLP3"], 0 , 0)

rule_vocab.create_rule("TerminalTransformLeftMinus1", ["SMLM"], ["TLM1"], 0 , 0)
rule_vocab.create_rule("TerminalTransformLeftMinus2", ["SMLM"], ["TLM2"], 0 , 0)
rule_vocab.create_rule("TerminalTransformLeftMinus3", ["SMLM"], ["TLM3"], 0 , 0)

rule_vocab.create_rule("TerminalEndLimb", ["EM"], ["U1"], 0 , 0)
rule_vocab.create_rule("TerminalJoint", ["J"], ["J1"], 0 , 0)


list_J = node_vocab.get_list_of_nodes(["J1"])
list_RM = node_vocab.get_list_of_nodes(["TR1", "TR2", "TR3","TRP1", "TRP2", "TRP3","TRM1", "TRM2", "TRM3"])
list_LM = node_vocab.get_list_of_nodes(["TL1", "TL2", "TL3","TLP1", "TLP2", "TLP3","TLM1", "TLM2", "TLM3"])
list_B = node_vocab.get_list_of_nodes(["L1", "L2", "L3", "F1", "F2", "F3"])

G = GraphGrammar()
"""
from utils.make_random_graph import make_random_graph

G = make_random_graph(20, rule_vocab, False)

plot_graph(G)
# %%
mysystem.Set_G_acc(chrono.ChVectorD(0,-1,0))

# Set solver settings
mysystem.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
mysystem.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
mysystem.SetMaxPenetrationRecoverySpeed(0.1)
mysystem.SetSolverMaxIterations(100)
mysystem.SetSolverForceTolerance(0)

my_robot = robot.Robot(G, mysystem)
joint_blocks = my_robot.get_joints

obj = chrono.ChBodyEasyCylinder(0.3,2, 1000,True,True,mat)
# obj = chrono.ChBodyEasySphere(0.5,1000,True,True,mat)
obj.SetCollide(True)
obj.SetMass(2)
obj.SetRot(chrono.ChQuaternionD(0.707, 0.707, 0, 0))
obj.SetPos(chrono.ChVectorD(0,0.55,0))

mysystem.Add(obj)
list_of_base_nodes = ["F1", "F2", "F3"]
for name in list_of_base_nodes:
    if len(my_robot.get_block_graph().find_nodes(rule_vocab.node_vocab.get_node(name))) > 0:
        base_id = my_robot.get_block_graph().find_nodes(rule_vocab.node_vocab.get_node(name))[0]

my_robot.block_map[base_id].body.SetBodyFixed(True)

# Make robot collide
blocks = my_robot.block_map.values()
body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
make_collide(body_block, CollisionGroup.Robot)
list_block = list(body_block)
force_list = []



# %%
T_0 = 1
T_1 = T_0

joint_blocks = my_robot.get_joints
print(joint_blocks)
const_trq = []
for joints in joint_blocks:
    i=0
    for joint in joints:
        const_trq.append(ctrl.ConstControl(joint, T_1*0.7*0.5**i))
        i+=1 


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

#partition_dfs = my_robot.get_dfs_partiton()

#J_NODES_NEW = nodes_division(robot, list_J)
#B_NODES_NEW = nodes_division(robot, list_B)
#LM_NODES_NEW = nodes_division(robot, list_LM)
#RM_NODES_NEW = nodes_division(robot, list_RM)
#abcd = []
# for i in range(len(J_NODES_NEW)):
#     abcd.append(J_NODES_NEW[i].id)
#abcd.append(filter(lambda x: x.id == 23, J_NODES_NEW))
    


#LB_NODES_NEW = sort_left_right(robot, list_LM, list_B)
#RB_NODES_NEW = sort_left_right(robot, list_RM, list_B)

#test = [[23] ,[0.1, 0.2, 0.3, 0.4]]

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
"""


max_numbers_rules = 20

# Create graph envirenments for algorithm (not gym)
env = env_graph.GraphEnvironment(G, rule_vocab, node_vocab,max_numbers_rules)

# Hyperparameters: increasing: > error reward, < search time
time_limit = 10000
iteration_limit = 20000

# Initilize MCTS
searcher = mcts.mcts(timeLimit=time_limit)
finish = False


reward_map_2 = {"J1": 1, "L1": 2,"L2": 2, "L3": 2 , "F1": 1, "F2": 1, "F3": 1, "U1": 1, "TR1": 4, "TR2": 4, "TR3": 4, "TRP1": 4, "TRP2": 4, 
"TRP3": 4, "TRM1": 4, "TRM2": 4, "TRM3": 4, "TL1": 4, "TL2": 4, "TL3": 4, "TLP1": 4, "TLP2": 4, "TLP3": 4, "TLM1": 4, "TLM2": 4, "TLM3": 4}
env.set_node_rewards(reward_map_2, Reward.complex)

# Search until finding terminal mechanism with desired reward
while not finish:
    print("---")
    action = searcher.search(initialState=env)
    finish, final_graph = env.step(action)

# Plot final graph
plt.figure()
nx.draw_networkx(final_graph, pos=nx.kamada_kawai_layout(final_graph, dim=2), node_size=800,
                 labels={n: final_graph.nodes[n]["Node"].label for n in final_graph})

plt.show()