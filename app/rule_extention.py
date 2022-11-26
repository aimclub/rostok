
from rostok.graph_grammar.node import BlockWrapper, ROOT
from rostok.graph_grammar import node_vocabulary, rule_vocabulary
from rostok.block_builder.node_render import *

import pychrono as chrono

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np




def plot_graph(graph):
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(graph, dim=2), node_size=800,
                    labels={n: graph.nodes[n]["Node"].label for n in G})
    #plt.figure()
    #nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800)
    plt.show()
    

# Define materials
mat = chrono.ChMaterialSurfaceNSC()
mysystem = chrono.ChSystemNSC() 
mat.SetFriction(0.5)

def init_extension_rules():
    # %% Bodies for extansions rules
    width = [0.5, 0.7, 1.]
    alpha = 45
    length_link = [0.4, 0.8, 1.]

    flat = list(map(lambda x: BlockWrapper(ChronoBody, width=x, length=0.05),
                    width))

    link = list(map(lambda x: BlockWrapper(ChronoBody, length=x),
                    length_link))


    u1 = BlockWrapper(ChronoBody, width=0.1, length=0.0)

    # %% Tranform for extansions rules 

    MOVE_TO_RIGHT_SIDE = map(lambda x: FrameTransform([x, 0, 0],[0,0,1,0]),
                    width)
    MOVE_TO_RIGHT_SIDE_PLUS = map(lambda x: FrameTransform([x, 0, +0.3],[0,0,1,0]),
                    width)
    MOVE_TO_RIGHT_SIDE_MINUS = map(lambda x: FrameTransform([x, 0, -0.3],[0,0,1,0]),
                    width)
    MOVE_TO_LEFT_SIDE = map(lambda x:  FrameTransform([-x, 0, 0],[1,0,0,0]),
                    width)
    MOVE_TO_LEFT_SIDE_PLUS = map(lambda x: FrameTransform([-x, 0, +0.3],[1,0,0,0]),
                    width)
    MOVE_TO_LEFT_SIDE_MINUS = map(lambda x: FrameTransform([-x, 0, -0.3],[1,0,0,0]),
                    width)

    quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
    ROTATE_TO_ALPHA = FrameTransform([0, 0, 0],[quat_Y_ang_alpha.e0,quat_Y_ang_alpha.e1,
                                            quat_Y_ang_alpha.e2,quat_Y_ang_alpha.e3])


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
    type_of_input = ChronoRevolveJoint.InputType.Torque

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
    list_B = node_vocab.get_list_of_nodes(["L1", "L2", "L3", "F1", "F2", "F3", "U1"])
    # Required for criteria calc
    node_features = [list_B, list_J, list_LM, list_RM]
    return rule_vocab, node_features