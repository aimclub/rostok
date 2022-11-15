# imports from our code
import context
from engine import rule_vocabulary
from engine import node_vocabulary
from engine.node import ROOT, GraphGrammar, BlockWrapper
import engine.robot as robot
import engine.control as ctrl
from engine.node_render import *
from utils.blocks_utils import make_collide, CollisionGroup   

# imports from standard libs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
# chrono imports
import pychrono as chrono
import pychrono.irrlicht as chronoirr
from pychrono import ChCoordsysD, ChVectorD, ChQuaternionD
from pychrono import Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X, \
    Q_ROTATE_Y_TO_X, Q_ROTATE_Y_TO_Z, \
    Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z



def plot_graph(graph:GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(graph, dim=2), node_size=800,
                    labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()

# Define block types
mat = chrono.ChMaterialSurfaceNSC()
mat.SetFriction(0.5)
mat.SetDampingF(0.1)

# Bodies
link1 = BlockWrapper(ChronoBody, length=0.3)
link2 = BlockWrapper(ChronoBody, length=0.2)

flat1 = BlockWrapper(ChronoBody, width=0.4, length=0.1)
flat2 = BlockWrapper(ChronoBody, width=0.7, length=0.1)

u1 = BlockWrapper(ChronoBody, width=0.1, length=0.1)

# Transforms
RZX = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_X)
RZY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_Y)
RXY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_X_TO_Y)

MOVE_ZX_PLUS = FrameTransform([0.3,0,0.3],[1,0,0,0])
MOVE_ZX_MINUS = FrameTransform([-0.3,0,-0.3],[1,0,0,0])

MOVE_X_PLUS = FrameTransform([0.3,0,0.],[1,0,0,0])
MOVE_Z_PLUS_X_MINUS = FrameTransform([-0.3,0,0.3],[1,0,0,0])

transform_rzx = BlockWrapper(ChronoTransform, RZX)
transform_rzy = BlockWrapper(ChronoTransform, RZY)
transform_rxy = BlockWrapper(ChronoTransform, RXY)
transform_mzx_plus = BlockWrapper(ChronoTransform, MOVE_ZX_PLUS)
transform_mzx_minus = BlockWrapper(ChronoTransform, MOVE_ZX_MINUS)
transform_mx_plus = BlockWrapper(ChronoTransform, MOVE_X_PLUS)
transform_mz_plus_x_minus = BlockWrapper(ChronoTransform, MOVE_Z_PLUS_X_MINUS)

type_of_input = ChronoRevolveJoint.InputType.Torque
# Joints
revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)


# %%
type_of_input = ChronoRevolveJoint.InputType.Position

# Joints
revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)

node_vocab = node_vocabulary.NodeVocabulary()
node_vocab.add_node(ROOT)
node_vocab.create_node("J")
node_vocab.create_node("L")
node_vocab.create_node("F")
node_vocab.create_node("M")
node_vocab.create_node("EF")
node_vocab.create_node("EM")

node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=revolve1)
node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=link1)
node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=link2)
node_vocab.create_node(label="F1", is_terminal=True, block_wrapper=flat1)
node_vocab.create_node(label="F2", is_terminal=True, block_wrapper=flat2)
node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=u1)
node_vocab.create_node(label="T1", is_terminal=True, block_wrapper=transform_mx_plus)
node_vocab.create_node(label="T2", is_terminal=True, block_wrapper=transform_mz_plus_x_minus)
node_vocab.create_node(label="T3", is_terminal=True, block_wrapper=transform_mzx_plus)
node_vocab.create_node(label="T4", is_terminal=True, block_wrapper=transform_mzx_minus)


rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

rule_vocab.create_rule("FlatCreate", ["ROOT"], ["F"], 0 , 0)
rule_vocab.create_rule("Mount", ["F"], ["F", "M", "EM"], 0 , 0, [(0,1), (1,2)])
rule_vocab.create_rule("MountAdd", ["M"], ["M", "EM"], 0 , 1, [(0,1)])
rule_vocab.create_rule("FingerUpper", ["EM"], ["J", "L", "EM"], 0 , 2, [(0,1), (1,2)])


rule_vocab.create_rule("TerminalFlat1", ["F"], ["F1"], 0 ,0)
rule_vocab.create_rule("TerminalFlat2", ["F"], ["F2"], 0 ,0)

rule_vocab.create_rule("TerminalL1", ["L"], ["L1"], 0 ,0)
rule_vocab.create_rule("TerminalL2", ["L"], ["L2"], 0 ,0)

rule_vocab.create_rule("TerminalTransformRX", ["M"], ["T1"], 0 ,0)
rule_vocab.create_rule("TerminalTransformLZ", ["M"], ["T2"], 0 ,0)
rule_vocab.create_rule("TerminalTransformR", ["M"], ["T3"], 0 ,0)
rule_vocab.create_rule("TerminalTransformL", ["M"], ["T4"], 0 ,0)
rule_vocab.create_rule("TerminalEndLimb", ["EM"], ["U1"], 0 ,0)
rule_vocab.create_rule("TerminalJoint", ["J"], ["J1"], 0 ,0)

rule_vocab.check_rules()

def make_random_graph(n_iter: int, rule_vocab: rule_vocabulary.RuleVocabulary, use_nonterminal_only: bool=True):
    G = GraphGrammar()
    for _ in range(n_iter//2+1):
        rules = rule_vocab.get_list_of_applicable_nonterminal_rules(G)
        if len(rules)>0:
            rule = rule_vocab.get_rule(rules[np.random.choice(len(rules))])
            G.apply_rule(rule)
        else: break
    
    for _ in range(n_iter-(n_iter//2+1)):
        if use_nonterminal_only:
            rules = rule_vocab.get_list_of_applicable_nonterminal_rules(G)
        else: rules = rule_vocab.get_list_of_applicable_rules(G)
        if len(rules)>0:
            rule = rule_vocab.get_rule(rules[np.random.choice(len(rules))])
            G.apply_rule(rule)
        else: break
    rule_vocab.make_graph_terminal(G)
    return G
    
G = make_random_graph(20, rule_vocab, False)
print(G.graph_partition_dfs())
plot_graph(G)


list_of_base_nodes = ["F1", "F2"]
mysystem = chrono.ChSystemNSC()
mysystem.Set_G_acc(chrono.ChVectorD(0,0,0))

my_robot = robot.Robot(G, mysystem)
joint_blocks = my_robot.get_joints[0]
obj = chrono.ChBodyEasyBox(0.2,0.2,0.6,1000,True,True,mat)
obj.SetCollide(True)
obj.SetPos(chrono.ChVectorD(0,1.2,0))
mysystem.Add(obj)
for name in list_of_base_nodes:
    if len(my_robot.get_block_graph().find_nodes(rule_vocab.node_vocab.get_node(name))) > 0:
        base_id = my_robot.get_block_graph().find_nodes(rule_vocab.node_vocab.get_node(name))[0]

my_robot.block_map[base_id].body.SetBodyFixed(True)
#des_points_1 = np.array([0, 0.1, 0.2, 0.3, 0.4])
pid_track = []
for joint in joint_blocks:
    pid_track.append(ctrl.ChControllerPID(joint ,80.,5.,1.)) # ctrl.TrackingControl(joint)
    #pid_track[-1].set_des_positions_interval(des_points_1,(0.1,2))

vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(mysystem)
vis.SetWindowSize(1024,768)
vis.SetWindowTitle('Custom contact demo')
vis.Initialize()
vis.AddCamera(chrono.ChVectorD(8, 8, -6))
vis.AddTypicalLights()
blocks = my_robot.block_map.values()
body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
make_collide(body_block, CollisionGroup.Robot)
#stopper = StopSimulation(mysystem, robot, obj, 1, 0.1)
# Create simulation loop
while vis.Run():
    mysystem.Update()
    mysystem.DoStepDynamics(5e-3)
    vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
    vis.Render()
    #if stopper.stop_simulation(): break
    vis.EndScene()