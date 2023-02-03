from rostok.graph_grammar.node import BlockWrapper, ROOT
from rostok.graph_grammar import node_vocabulary, rule_vocabulary
from rostok.block_builder.node_render import *


import pychrono as chrono

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def create_rules():
    width = [0.1, 0.35, 0.5]
    length_link = [0.4, 0.6, 0.8]
    flat = list(map(lambda x: BlockWrapper(FlatChronoBody, width_x=x, height_y=0.05, depth_z=0.8),
                    width))

    link = list(map(lambda x: BlockWrapper(LinkChronoBody, length_y=x),
                    length_link))


    u1 = BlockWrapper(MountChronoBody, width_x=0.1, length_y=0.05)
    u2 = BlockWrapper(MountChronoBody, width_x=0.2, length_y=0.1)
    MOVE_TO_RIGHT_SIDE = map(lambda x: FrameTransform([x*10, 0, 0], [0,0,1,0]), width)
    def rotation_y(alpha):
        quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
        return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2,quat_Y_ang_alpha.e3]
    
    def rotation_z(alpha):
        quat_Z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
        return [quat_Z_ang_alpha.e0, quat_Z_ang_alpha.e1, quat_Z_ang_alpha.e2,quat_Z_ang_alpha.e3]
    #MOVE = FrameTransform([1/2, 0, 1/2*3**0.5], rotation_y(60))
    MOVE = FrameTransform([0.05, -0.3, 0], rotation_z(90))
    #MOVE = FrameTransform([1, 0, 0], rotation(45))
    transform_to_right_mount = list(map(lambda x: BlockWrapper(ChronoTransform, x),
                    MOVE_TO_RIGHT_SIDE))
    round_transform = BlockWrapper(ChronoTransform, MOVE)
    type_of_input = ChronoRevolveJoint.InputType.TORQUE
    revolve = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)
    # Nodes
    node_vocab = node_vocabulary.NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node("F")
    node_vocab.create_node("T")
    node_vocab.create_node("J")
    node_vocab.create_node("L")
    node_vocab.create_node("M")
    node_vocab.create_node("FS")
    node_vocab.create_node("EM")

    node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=link[0])
    node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=link[1])
    node_vocab.create_node(label="L3", is_terminal=True, block_wrapper=link[2])
    node_vocab.create_node(label="F1", is_terminal=True, block_wrapper=flat[0])
    node_vocab.create_node(label="F2", is_terminal=True, block_wrapper=flat[1])
    node_vocab.create_node(label="F3", is_terminal=True, block_wrapper=flat[2])
    node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=u1)
    node_vocab.create_node(label="U2", is_terminal=True, block_wrapper=u2)

    node_vocab.create_node(label="ET1", is_terminal=True, block_wrapper=transform_to_right_mount[0])
    node_vocab.create_node(label="T1", is_terminal=True, block_wrapper=round_transform)


    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

    # rule_vocab.create_rule("InitMechanism", ["ROOT"], ["F", "T","FS"], 0 , 0, [(0,1),(1,2)])
    # rule_vocab.create_rule("Add_First_Mount", ["T"], ["T", "M","FS"], 0 , 0, [(0,1),(1,2)])
    # rule_vocab.create_rule("Add_Mount", ["M"], ["M", "M", "FS"], 0 , 0,[(0,1),(1, 2)])
    # rule_vocab.create_rule("FirstLink", ["FS"], ["U1","J1", "L","EM"], 0 , 3, [(0,1),(1, 2), (2,3)])
    # rule_vocab.create_rule("FingerUpper", ["EM"], ["J1", "L","EM"], 0 , 2, [(0,1),(1, 2)])

    rule_vocab.create_rule("InitMechanism", ["ROOT"], ["F1", "T","EM"], 0 , 0, [(0,1),(1,2)])
    rule_vocab.create_rule("Add_First_Mount", ["T"], ["T", "M","EM"], 0 , 0, [(0,1),(1,2)])
    rule_vocab.create_rule("Add_Mount", ["M"], ["T1", "M", "EM"], 0 , 0,[(0,1),(1, 2)])
    #rule_vocab.create_rule("FirstLink", ["FS"], ["U1","J1", "L","EM"], 0 , 3, [(0,1),(1, 2), (2,3)])
    rule_vocab.create_rule("FingerUpper", ["EM"], ["J1", "L","EM"], 0 , 2, [(0,1),(1, 2)])
    rule_vocab.create_rule("FingerSplitter", ["L"], ["L", "M", "L", "EM"], 0 , 0, [(0, 1), (1, 2), (2, 3)])
    rule_vocab.create_rule("DoubleFinger", ["EM"], ["L", "M", "EM"], 0 , 0, [(0, 1), (1, 2)])

    rule_vocab.create_rule("TerminalFlat1", ["F"], ["F1"], 0 , 0)
    #rule_vocab.create_rule("TerminalFlat2", ["F"], ["F2"], 0 , 0)
    #rule_vocab.create_rule("TerminalFlat3", ["F"], ["F3"], 0 , 0)

    #rule_vocab.create_rule("TerminalL1", ["L"], ["L1"], 0 , 0)
    rule_vocab.create_rule("TerminalL2", ["L"], ["L2"], 0 , 0)
    #rule_vocab.create_rule("TerminalL3", ["L"], ["L3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformRight1", ["T"], ["ET1"], 0 , 0)
    rule_vocab.create_rule("TerminalRoundTransform", ["M"], ["T1"], 0 , 0)


    rule_vocab.create_rule("TerminalEndLimb1", ["EM"], ["U1"], 0 , 0)
    #rule_vocab.create_rule("FingerStartTerminal", ["FS"], ["U1"], 0 , 0)
    #rule_vocab.create_rule("TerminalEndLimb2", ["EM"], ["U2"], 0 , 0)
    #rule_vocab.create_rule("TerminalJoint", ["J"], ["J1"], 0 , 0)


    list_J = node_vocab.get_list_of_nodes(["J1"])
    list_RM = node_vocab.get_list_of_nodes(["ET1", "T1"])
    list_LM = node_vocab.get_list_of_nodes([])
    list_B = node_vocab.get_list_of_nodes(["L1", "L2", "L3", "F1", "F2", "F3", "U1", "U2"])
    # Required for criteria calc
    node_features = [list_B, list_J, list_LM, list_RM]
    return rule_vocab, node_features
