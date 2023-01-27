from rostok.graph_grammar.node import BlockWrapper, ROOT
from rostok.graph_grammar import node_vocabulary, rule_vocabulary
from rostok.block_builder.node_render import *
import rostok.intexp as intexp
from pickup_pipes_utils import get_main_axis_pipe

import pychrono as chrono

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def create_rules_to_pickup_pipe(path_to_pipe_obj, path_to_pipe_xml):
    with_angle = False
    COEFFICIENTS = np.array([2, 3, 5])

    obj_db = intexp.entity.TesteeObject()
    obj_db.load_object_mesh(path_to_pipe_obj)
    obj_db.load_object_description(path_to_pipe_xml)
    long_dimension, short_dimension, axis = get_main_axis_pipe(obj_db)

    start_point_grasp = BlockWrapper(FlatChronoBody,
                                     height_y=short_dimension[1] * 0.01,
                                     width_x=short_dimension[1] * 0.01,
                                     depth_z=short_dimension[1] * 0.01,
                                     is_collide=False)

    flat = BlockWrapper(FlatChronoBody,
                        width_x=short_dimension[1],
                        depth_z=long_dimension[1] * 0.6,
                        height_y=short_dimension[1] * 0.5)

    link = list(
        map(lambda x: BlockWrapper(LinkChronoBody, length_y=x, width_x= x/7.5, depth_z =x/2), COEFFICIENTS * short_dimension[1]))

    u1 = BlockWrapper(MountChronoBody, width_x=short_dimension[1]/1.5, depth_z=short_dimension[1]/1.5)
    u2 = BlockWrapper(MountChronoBody, width_x=short_dimension[1]/2, depth_z=short_dimension[1]/1.5)

    def rotation(alpha):
        quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
        return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]

    MOVE_TO_RIGHT_GRASP = FrameTransform([0, 0, +long_dimension[1] * 0.25], [0, 0, 1, 0])

    MOVE_TO_LEFT_GRASP = FrameTransform([0, 0, -long_dimension[1] * 0.25], [0, 0, 1, 0])

    MOVE_TO_CENTER_GRASP = FrameTransform([0, 0, 0], [0, 0, 1, 0])

    MOVE_TO_RIGHT_SIDE = FrameTransform([short_dimension[1], 0, 0], [0, 0, 1, 0])

    MOVE_TO_RIGHT_SIDE_PLUS = FrameTransform([short_dimension[1], 0, +0.3], [0, 0, 1, 0])

    MOVE_TO_RIGHT_SIDE_PLUS_ANGLE = FrameTransform([short_dimension[1], 0, +0.3], rotation(150))

    MOVE_TO_RIGHT_SIDE_MINUS = FrameTransform([short_dimension[1], 0, -0.3], [0, 0, 1, 0])

    MOVE_TO_RIGHT_SIDE_MINUS_ANGLE = FrameTransform([short_dimension[1], 0, -0.3], rotation(210))

    MOVE_TO_LEFT_SIDE = FrameTransform([-short_dimension[1], 0, 0], [1, 0, 0, 0])

    MOVE_TO_LEFT_SIDE_PLUS = FrameTransform([-short_dimension[1], 0, +0.3], [1, 0, 0, 0])
    MOVE_TO_LEFT_SIDE_PLUS_ANGLE = FrameTransform([-short_dimension[1], 0, +0.3], rotation(30))

    MOVE_TO_LEFT_SIDE_MINUS = FrameTransform([-short_dimension[1], 0, -0.3], [1, 0, 0, 0])
    MOVE_TO_LEFT_SIDE_MINUS_ANGLE = FrameTransform([-short_dimension[1], 0, -0.3], rotation(-30))

    transform_to_right_mechanims = BlockWrapper(ChronoTransform, MOVE_TO_RIGHT_GRASP)

    transform_to_left_mechanims = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_GRASP)

    transform_to_center_mechanims = BlockWrapper(ChronoTransform, MOVE_TO_CENTER_GRASP)

    transform_to_right_mount = BlockWrapper(ChronoTransform, MOVE_TO_RIGHT_SIDE)

    transform_to_right_mount_plus = BlockWrapper(ChronoTransform, MOVE_TO_RIGHT_SIDE_PLUS)

    transform_to_right_mount_plus_angle = BlockWrapper(ChronoTransform,
                                                       MOVE_TO_RIGHT_SIDE_PLUS_ANGLE)

    transform_to_right_mount_minus = BlockWrapper(ChronoTransform, MOVE_TO_RIGHT_SIDE_MINUS)

    transform_to_right_mount_minus_angle = BlockWrapper(ChronoTransform,
                                                        MOVE_TO_RIGHT_SIDE_MINUS_ANGLE)

    transform_to_left_mount = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_SIDE)

    transform_to_left_mount_plus = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_SIDE_PLUS)

    transform_to_left_mount_plus_angle = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_SIDE_PLUS_ANGLE)

    transform_to_left_mount_minus = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_SIDE_MINUS)

    transform_to_left_mount_minus_angle = BlockWrapper(ChronoTransform,
                                                       MOVE_TO_LEFT_SIDE_MINUS_ANGLE)

    # %%
    type_of_input = ChronoRevolveJoint.InputType.TORQUE

    # Joints
    revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z, type_of_input,
                            stiffness = 900, damping = 50, equilibrium_position = -chrono.CH_C_PI / 3)

    # Nodes
    node_vocab = node_vocabulary.NodeVocabulary()
    node_vocab.add_node(ROOT)

    node_vocab.create_node("GL")  # Left grasp mechanism
    node_vocab.create_node("GC")  # Center grasp mechanism
    node_vocab.create_node("GR")  # Right grasp mechanism
    
    node_vocab.create_node("GL1")  # Initilized left grasp mechanism
    node_vocab.create_node("GC1")  # Initilized center grasp mechanism
    node_vocab.create_node("GR1")  # Initilized right grasp mechanism

    node_vocab.create_node("L")
    node_vocab.create_node("F")
    node_vocab.create_node("M")
    node_vocab.create_node("EF")
    node_vocab.create_node("EM")
    node_vocab.create_node("SML")
    node_vocab.create_node("SMR")
    node_vocab.create_node("SMRP")
    node_vocab.create_node("SMRPA")
    node_vocab.create_node("SMLP")
    node_vocab.create_node("SMLPA")
    node_vocab.create_node("SMRM")
    node_vocab.create_node("SMRMA")
    node_vocab.create_node("SMLM")
    node_vocab.create_node("SMLMA")

    #O = Node("O")

    node_vocab.create_node(label="SPG1", is_terminal=True, block_wrapper=start_point_grasp)

    node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=revolve1)

    node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=link[0])
    node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=link[1])
    node_vocab.create_node(label="L3", is_terminal=True, block_wrapper=link[2])
    node_vocab.create_node(label="F1", is_terminal=True, block_wrapper=flat)
    node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=u1)
    node_vocab.create_node(label="U2", is_terminal=True, block_wrapper=u2)

    node_vocab.create_node(label="TGR1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mechanims)

    node_vocab.create_node(label="TGL1",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mechanims)

    node_vocab.create_node(label="TGC1",
                           is_terminal=True,
                           block_wrapper=transform_to_center_mechanims)

    node_vocab.create_node(label="TR1", is_terminal=True, block_wrapper=transform_to_right_mount)

    node_vocab.create_node(label="TRP1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_plus)

    node_vocab.create_node(label="TRM1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_minus)


    node_vocab.create_node(label="TL1", is_terminal=True, block_wrapper=transform_to_left_mount)

    node_vocab.create_node(label="TLP1",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_plus)

    node_vocab.create_node(label="TLM1",
                        is_terminal=True,
                        block_wrapper=transform_to_left_mount_minus)

    node_vocab.create_node(label="TRPA1",
                        is_terminal=True,
                        block_wrapper=transform_to_right_mount_plus_angle)
    node_vocab.create_node(label="TRMA1",
                        is_terminal=True,
                        block_wrapper=transform_to_right_mount_minus_angle)
    node_vocab.create_node(label="TLPA1",
                        is_terminal=True,
                        block_wrapper=transform_to_left_mount_plus_angle)
    node_vocab.create_node(label="TLMA1",
                        is_terminal=True,
                        block_wrapper=transform_to_left_mount_minus_angle)



    # Defines rules
    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

    # Define rules of initilize topology grasp mechanisms
    rule_vocab.create_rule("InitilizeGrab_1", ["ROOT"], [
        "F",
        "GL",
        "GR",
    ], 0, 0, [(0, 1), (0, 2)])
    rule_vocab.create_rule("InitilizeGrab_2", ["ROOT"], ["F", "GL", "GR", "GC"], 0, 0, [(0, 1),
                                                                                        (0, 2),
                                                                                        (0, 3)])

    # Define rules of grasp mechanism
    grasp_mechanism_nodes = ('GL', 'GR', 'GC')
    initilized_grasp_mechanism_nodes = ('GL1', 'GR1', 'GC1')
    for idx, grasp_node in enumerate(grasp_mechanism_nodes):
        name_init_mechanism = grasp_node + '_InitMechanism_'

        # rule_vocab.create_rule(name_init_mechanism + '2', [grasp_node],
        #                        [initilized_grasp_mechanism_nodes[idx], "SML", "SMR", "EM", "EM"], 0,
        #                        0, [(0, 1), (0, 2), (1, 3), (2, 4)])

        rule_vocab.create_rule(
            name_init_mechanism + '3_R', [grasp_node],
            [initilized_grasp_mechanism_nodes[idx], "SML", "SMRP", "SMRM", "EM", "EM", "EM"], 0, 0,
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)])
        rule_vocab.create_rule(
            name_init_mechanism + '3_L', [grasp_node],
            [initilized_grasp_mechanism_nodes[idx], "SMLP", "SMLM", "SMR", "EM", "EM", "EM"], 0, 0,
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)])
        if with_angle:
            rule_vocab.create_rule(
                name_init_mechanism + '3_R_A', [grasp_node],
                [initilized_grasp_mechanism_nodes[idx], "SML", "SMRPA", "SMRMA", "EM", "EM", "EM"], 0,
                0, [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)])
            rule_vocab.create_rule(
                name_init_mechanism + '3_L_A', [grasp_node],
                [initilized_grasp_mechanism_nodes[idx], "SMLPA", "SMLMA", "SMR", "EM", "EM", "EM"], 0,
                0, [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)])
        rule_vocab.create_rule(name_init_mechanism + '4_A', [grasp_node], [
            initilized_grasp_mechanism_nodes[idx], "SMLPA", "SMLMA", "SMRPA", "SMRMA", "EM", "EM",
            "EM", "EM"
        ], 0, 0, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8)])
        # rule_vocab.create_rule(name_init_mechanism + '4', [grasp_node], [
        #     initilized_grasp_mechanism_nodes[idx], "SMLP", "SMLM", "SMRP", "SMRM", "EM", "EM", "EM",
        #     "EM"
        # ], 0, 0, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8)])

    rule_vocab.create_rule("TerminalStartRightMechanism", ['GR1'], ['TGR1', 'SPG1'], 0, 1, [(0, 1)])
    rule_vocab.create_rule("TerminalStartLeftMechanism", ['GL1'], ['TGL1', 'SPG1'], 0, 1, [(0, 1)])
    rule_vocab.create_rule("TerminalStartCenterMechanism", ['GC1'], ['TGC1', 'SPG1'], 0, 1,
                           [(0, 1)])

    rule_vocab.create_rule("FingerUpper", ["EM"], ["J1", "L", "EM"], 0, 2, [(0, 1), (1, 2)])

    rule_vocab.create_rule("TerminalFlat1", ["F"], ["F1"], 0, 0)

    rule_vocab.create_rule("TerminalL1", ["L"], ["L1"], 0, 0)

    rule_vocab.create_rule("TerminalTransformRight1", ["SMR"], ["TR1"], 0, 0)

    rule_vocab.create_rule("TerminalTransformRightPlus1", ["SMRP"], ["TRP1"], 0, 0)


    rule_vocab.create_rule("TerminalTransformRightMinus1", ["SMRM"], ["TRM1"], 0, 0)


    rule_vocab.create_rule("TerminalTransformLeft1", ["SML"], ["TL1"], 0, 0)

    rule_vocab.create_rule("TerminalTransformLeftPlus1", ["SMLP"], ["TLP1"], 0, 0)

    rule_vocab.create_rule("TerminalTransformLeftMinus1", ["SMLM"], ["TLM1"], 0, 0)

    
    rule_vocab.create_rule("TerminalTransformRightPlusAngle1", ["SMRPA"], ["TRPA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightMinusAngle1", ["SMRMA"], ["TRMA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftPlusAngle1", ["SMLPA"], ["TLPA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftMinusAngle1", ["SMLMA"], ["TLMA1"], 0, 0)

    rule_vocab.create_rule("TerminalEndLimb1", ["EM"], ["U1"], 0, 0)
    rule_vocab.create_rule("TerminalEndLimb2", ["EM"], ["U2"], 0, 0)

    list_J = node_vocab.get_list_of_nodes(["J1"])

    list_RM = node_vocab.get_list_of_nodes(["TR1", "TRP1", "TRM1", "TRPA1", "TRMA1"])
    list_LM = node_vocab.get_list_of_nodes(["TL1", "TLP1", "TLM1", "TLPA1", "TLMA1"])
        
    list_B = node_vocab.get_list_of_nodes(["L1", "L2", "L3", "F1", "U1", "U2"])
    # Required for criteria calc
    node_features = [list_B, list_J, list_LM, list_RM]
    return rule_vocab, node_features

def create_rules_to_pickup_pipe_ver_2(path_to_pipe_obj, path_to_pipe_xml):
    with_angle = False
    COEFFICIENTS = np.array([2, 3, 5])

    obj_db = intexp.entity.TesteeObject()
    obj_db.load_object_mesh(path_to_pipe_obj)
    obj_db.load_object_description(path_to_pipe_xml)
    long_dimension, short_dimension, __ = get_main_axis_pipe(obj_db)

    start_point_grasp = BlockWrapper(FlatChronoBody,
                                     height_y=short_dimension[1] * 0.01,
                                     width_x=short_dimension[1] * 0.01,
                                     depth_z=short_dimension[1] * 0.01,
                                     is_collide=False)

    flat = BlockWrapper(FlatChronoBody,
                        width_x=short_dimension[1],
                        depth_z=long_dimension[1] * 0.6,
                        height_y=short_dimension[1] * 0.5)

    link = list(
        map(lambda x: BlockWrapper(LinkChronoBody, length_y=x, width_x= x/7.5, depth_z =x/2), COEFFICIENTS * short_dimension[1]))

    u1 = BlockWrapper(MountChronoBody, width_x=short_dimension[1]/1.5, depth_z=short_dimension[1]/1.5)
    u2 = BlockWrapper(MountChronoBody, width_x=short_dimension[1]/2, depth_z=short_dimension[1]/1.5)

    def rotation(alpha):
        quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
        return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]

    MOVE_TO_RIGHT_GRASP = FrameTransform([0, 0, +long_dimension[1] * 0.25], [0, 0, 1, 0])

    MOVE_TO_LEFT_GRASP = FrameTransform([0, 0, -long_dimension[1] * 0.25], [0, 0, 1, 0])

    MOVE_TO_CENTER_GRASP = FrameTransform([0, 0, 0], [0, 0, 1, 0])

    MOVE_TO_RIGHT_SIDE = FrameTransform([short_dimension[1], 0, 0], [0, 0, 1, 0])

    MOVE_TO_RIGHT_SIDE_PLUS = FrameTransform([short_dimension[1], 0, +0.3], [0, 0, 1, 0])

    MOVE_TO_RIGHT_SIDE_PLUS_ANGLE = FrameTransform([short_dimension[1], 0, +0.3], rotation(150))

    MOVE_TO_RIGHT_SIDE_MINUS = FrameTransform([short_dimension[1], 0, -0.3], [0, 0, 1, 0])

    MOVE_TO_RIGHT_SIDE_MINUS_ANGLE = FrameTransform([short_dimension[1], 0, -0.3], rotation(210))

    MOVE_TO_LEFT_SIDE = FrameTransform([-short_dimension[1], 0, 0], [1, 0, 0, 0])

    MOVE_TO_LEFT_SIDE_PLUS = FrameTransform([-short_dimension[1], 0, +0.3], [1, 0, 0, 0])
    MOVE_TO_LEFT_SIDE_PLUS_ANGLE = FrameTransform([-short_dimension[1], 0, +0.3], rotation(30))

    MOVE_TO_LEFT_SIDE_MINUS = FrameTransform([-short_dimension[1], 0, -0.3], [1, 0, 0, 0])
    MOVE_TO_LEFT_SIDE_MINUS_ANGLE = FrameTransform([-short_dimension[1], 0, -0.3], rotation(-30))

    transform_to_right_mechanims = BlockWrapper(ChronoTransform, MOVE_TO_RIGHT_GRASP)

    transform_to_left_mechanims = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_GRASP)

    transform_to_center_mechanims = BlockWrapper(ChronoTransform, MOVE_TO_CENTER_GRASP)

    transform_to_right_mount = BlockWrapper(ChronoTransform, MOVE_TO_RIGHT_SIDE)

    transform_to_right_mount_plus = BlockWrapper(ChronoTransform, MOVE_TO_RIGHT_SIDE_PLUS)

    transform_to_right_mount_plus_angle = BlockWrapper(ChronoTransform,
                                                       MOVE_TO_RIGHT_SIDE_PLUS_ANGLE)

    transform_to_right_mount_minus = BlockWrapper(ChronoTransform, MOVE_TO_RIGHT_SIDE_MINUS)

    transform_to_right_mount_minus_angle = BlockWrapper(ChronoTransform,
                                                        MOVE_TO_RIGHT_SIDE_MINUS_ANGLE)

    transform_to_left_mount = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_SIDE)

    transform_to_left_mount_plus = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_SIDE_PLUS)

    transform_to_left_mount_plus_angle = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_SIDE_PLUS_ANGLE)

    transform_to_left_mount_minus = BlockWrapper(ChronoTransform, MOVE_TO_LEFT_SIDE_MINUS)

    transform_to_left_mount_minus_angle = BlockWrapper(ChronoTransform,
                                                       MOVE_TO_LEFT_SIDE_MINUS_ANGLE)

    # %%
    type_of_input = ChronoRevolveJoint.InputType.TORQUE

    # Joints
    revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z, type_of_input,
                            stiffness = 900, damping = 50, equilibrium_position = -chrono.CH_C_PI / 3)

    # Nodes
    node_vocab = node_vocabulary.NodeVocabulary()
    node_vocab.add_node(ROOT)

    node_vocab.create_node("GL")  # Left grasp mechanism
    node_vocab.create_node("GC")  # Center grasp mechanism
    node_vocab.create_node("GR")  # Right grasp mechanism
    
    node_vocab.create_node("GL1")  # Initilized left grasp mechanism
    node_vocab.create_node("GC1")  # Initilized center grasp mechanism
    node_vocab.create_node("GR1")  # Initilized right grasp mechanism

    node_vocab.create_node("L")
    node_vocab.create_node("F")
    node_vocab.create_node("M")
    node_vocab.create_node("EF")
    node_vocab.create_node("EM")
    node_vocab.create_node("SML")
    node_vocab.create_node("SMR")
    node_vocab.create_node("SMRP")
    node_vocab.create_node("SMRPA")
    node_vocab.create_node("SMLP")
    node_vocab.create_node("SMLPA")
    node_vocab.create_node("SMRM")
    node_vocab.create_node("SMRMA")
    node_vocab.create_node("SMLM")
    node_vocab.create_node("SMLMA")

    #O = Node("O")

    node_vocab.create_node(label="SPG1", is_terminal=True, block_wrapper=start_point_grasp)

    node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=revolve1)

    node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=link[0])
    node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=link[1])
    node_vocab.create_node(label="L3", is_terminal=True, block_wrapper=link[2])
    node_vocab.create_node(label="F1", is_terminal=True, block_wrapper=flat)
    node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=u1)
    node_vocab.create_node(label="U2", is_terminal=True, block_wrapper=u2)

    node_vocab.create_node(label="TGR1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mechanims)

    node_vocab.create_node(label="TGL1",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mechanims)

    node_vocab.create_node(label="TGC1",
                           is_terminal=True,
                           block_wrapper=transform_to_center_mechanims)

    node_vocab.create_node(label="TR1", is_terminal=True, block_wrapper=transform_to_right_mount)

    node_vocab.create_node(label="TRP1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_plus)

    node_vocab.create_node(label="TRM1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_minus)


    node_vocab.create_node(label="TL1", is_terminal=True, block_wrapper=transform_to_left_mount)

    node_vocab.create_node(label="TLP1",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_plus)

    node_vocab.create_node(label="TLM1",
                        is_terminal=True,
                        block_wrapper=transform_to_left_mount_minus)

    node_vocab.create_node(label="TRPA1",
                        is_terminal=True,
                        block_wrapper=transform_to_right_mount_plus_angle)
    node_vocab.create_node(label="TRMA1",
                        is_terminal=True,
                        block_wrapper=transform_to_right_mount_minus_angle)
    node_vocab.create_node(label="TLPA1",
                        is_terminal=True,
                        block_wrapper=transform_to_left_mount_plus_angle)
    node_vocab.create_node(label="TLMA1",
                        is_terminal=True,
                        block_wrapper=transform_to_left_mount_minus_angle)



    # Defines rules
    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

    # Define rules of initilize topology grasp mechanisms
    rule_vocab.create_rule("InitilizeGrab_1", ["ROOT"], [
        "F",
        "GL",
        "GR",
    ], 0, 0, [(0, 1), (0, 2)])
    rule_vocab.create_rule("InitilizeGrab_2", ["ROOT"], ["F1", "GL", "GR", "GC"], 0, 0, [(0, 1),
                                                                                        (0, 2),
                                                                                        (0, 3)])

    # Define rules of grasp mechanism
    grasp_mechanism_nodes = ('GL', 'GR', 'GC')
    initilized_grasp_mechanism_nodes = ('GL1', 'GR1', 'GC1')
    for idx, grasp_node in enumerate(grasp_mechanism_nodes):
        name_init_mechanism = grasp_node + '_InitMechanism_'
        
        rule_vocab.create_rule(
            name_init_mechanism + '3_R', [grasp_node],
            [initilized_grasp_mechanism_nodes[idx], "SML", "SMRP", "SMRM", "EM", "EM", "EM"], 0, 0,
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)])
        rule_vocab.create_rule(
            name_init_mechanism + '3_L', [grasp_node],
            [initilized_grasp_mechanism_nodes[idx], "SMLP", "SMLM", "SMR", "EM", "EM", "EM"], 0, 0,
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)])
        if with_angle:
            rule_vocab.create_rule(
                name_init_mechanism + '3_R_A', [grasp_node],
                [initilized_grasp_mechanism_nodes[idx], "SML", "SMRPA", "SMRMA", "EM", "EM", "EM"], 0,
                0, [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)])
            rule_vocab.create_rule(
                name_init_mechanism + '3_L_A', [grasp_node],
                [initilized_grasp_mechanism_nodes[idx], "SMLPA", "SMLMA", "SMR", "EM", "EM", "EM"], 0,
                0, [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)])

    rule_vocab.create_rule("TerminalStartRightMechanism", ['GR1'], ['TGR1', 'SPG1'], 0, 1, [(0, 1)])
    rule_vocab.create_rule("TerminalStartLeftMechanism", ['GL1'], ['TGL1', 'SPG1'], 0, 1, [(0, 1)])
    rule_vocab.create_rule("TerminalStartCenterMechanism", ['GC1'], ['TGC1', 'SPG1'], 0, 1,
                           [(0, 1)])

    for idx, link in enumerate(("L1", "L2", "L3")):
        rule_vocab.create_rule("FingerUpper" + str(idx), ["EM"], ["J1", link, "EM"], 0, 2, [(0, 1), (1, 2)])


    rule_vocab.create_rule("TerminalTransformRight1", ["SMR"], ["TR1"], 0, 0)

    rule_vocab.create_rule("TerminalTransformRightPlus1", ["SMRP"], ["TRP1"], 0, 0)


    rule_vocab.create_rule("TerminalTransformRightMinus1", ["SMRM"], ["TRM1"], 0, 0)


    rule_vocab.create_rule("TerminalTransformLeft1", ["SML"], ["TL1"], 0, 0)

    rule_vocab.create_rule("TerminalTransformLeftPlus1", ["SMLP"], ["TLP1"], 0, 0)

    rule_vocab.create_rule("TerminalTransformLeftMinus1", ["SMLM"], ["TLM1"], 0, 0)

    
    rule_vocab.create_rule("TerminalTransformRightPlusAngle1", ["SMRPA"], ["TRPA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightMinusAngle1", ["SMRMA"], ["TRMA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftPlusAngle1", ["SMLPA"], ["TLPA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftMinusAngle1", ["SMLMA"], ["TLMA1"], 0, 0)

    rule_vocab.create_rule("TerminalEndLimb1", ["EM"], ["U1"], 0, 0)
    rule_vocab.create_rule("TerminalEndLimb2", ["EM"], ["U2"], 0, 0)

    list_J = node_vocab.get_list_of_nodes(["J1"])

    list_RM = node_vocab.get_list_of_nodes(["TR1", "TRP1", "TRM1", "TRPA1", "TRMA1"])
    list_LM = node_vocab.get_list_of_nodes(["TL1", "TLP1", "TLM1", "TLPA1", "TLMA1"])
        
    list_B = node_vocab.get_list_of_nodes(["L1", "L2", "L3", "F1", "U1", "U2"])
    # Required for criteria calc
    node_features = [list_B, list_J, list_LM, list_RM]
    return rule_vocab, node_features
