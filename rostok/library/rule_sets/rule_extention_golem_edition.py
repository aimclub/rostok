import numpy as np
import pychrono as chrono

from rostok.block_builder_chrono.block_classes import (ChronoEasyShape,
                                                       ChronoRevolveJoint,
                                                       ChronoTransform,
                                                       UniversalBox)
from rostok.block_builder_chrono.blocks_utils import FrameTransform
from rostok.graph import node_vocabulary
from rostok.graph.graph import ROOT
from rostok.graph.node import BlockWrapper
from rostok.graph_grammar import rule_vocabulary

# %% Bodies for extansions rules
width = [0.5, 0.6, 0.65]
alpha = 45
alpha_left = [0, 30, 60]
alpha_right = [180, 150, 120]
length_link = [0.3, 0.6, 0.8, 1]

flat = list(
    map(lambda x: BlockWrapper(UniversalBox, x=x, y=0.05, z=0.8), width))

link = list(map(lambda x: BlockWrapper(UniversalBox, x=0.1, y=x, z=0.3), length_link))

u1 = BlockWrapper(UniversalBox, x=0.1, y=0.05, z=0.3)
u2 = BlockWrapper(UniversalBox, x=0.2, y=0.1, z= 0.3)

# %% Tranform for extansions rules
# z_shift = [-0.3, 0, 0.3]
# MOVE_TO_RIGHT_SIDE = []
# for i in width:
#     for j in z_shift:
#         for alpha in alpha_right:
#             quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
#             ROTATE_TO_ALPHA = FrameTransform([i, 0, j],[quat_Y_ang_alpha.e0,quat_Y_ang_alpha.e1,
#                                         quat_Y_ang_alpha.e2,quat_Y_ang_alpha.e3])
#             MOVE_TO_RIGHT_SIDE.append(ROTATE_TO_ALPHA)

# transform_to_right_mount_list = list(map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE))


def rotation(alpha):
    quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
    return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]

MOVES_R = 0.4
MOVE_TO_RIGHT_SIDE = map(lambda x: FrameTransform([x, 0, 0], [0, 0, 1, 0]), width)
MOVE_TO_RIGHT_SIDE_PLUS = map(lambda x: FrameTransform([x, 0, +MOVES_R], [0, 0, 1, 0]), width)
MOVE_TO_RIGHT_SIDE_PLUS_ANGLE = map(lambda x: FrameTransform([x, 0, +MOVES_R], rotation(150)), width)
MOVE_TO_RIGHT_SIDE_MINUS = map(lambda x: FrameTransform([x, 0, -MOVES_R], [0, 0, 1, 0]), width)
MOVE_TO_RIGHT_SIDE_MINUS_ANGLE = map(lambda x: FrameTransform([x, 0, -MOVES_R], rotation(210)), width)
MOVE_TO_LEFT_SIDE = map(lambda x: FrameTransform([-x, 0, 0], [1, 0, 0, 0]), width)
MOVE_TO_LEFT_SIDE_PLUS = map(lambda x: FrameTransform([-x, 0, +MOVES_R], [1, 0, 0, 0]), width)
MOVE_TO_LEFT_SIDE_PLUS_ANGLE = map(lambda x: FrameTransform([-x, 0, +MOVES_R], rotation(30)), width)
MOVE_TO_LEFT_SIDE_MINUS = map(lambda x: FrameTransform([-x, 0, -MOVES_R], [1, 0, 0, 0]), width)
MOVE_TO_LEFT_SIDE_MINUS_ANGLE = map(lambda x: FrameTransform([-x, 0, -MOVES_R], rotation(-30)), width)

# quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
# ROTATE_TO_ALPHA = FrameTransform([0, 0, 0],[quat_Y_ang_alpha.e0,quat_Y_ang_alpha.e1,
#                                         quat_Y_ang_alpha.e2,quat_Y_ang_alpha.e3])

transform_to_right_mount = list(map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE))
transform_to_right_mount_plus = list(
    map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE_PLUS))
transform_to_right_mount_plus_angle = list(
    map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE_PLUS_ANGLE))
transform_to_right_mount_minus = list(
    map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE_MINUS))
transform_to_right_mount_minus_angle = list(
    map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE_MINUS_ANGLE))
transform_to_left_mount = list(map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE))
transform_to_left_mount_plus = list(
    map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE_PLUS))
transform_to_left_mount_plus_angle = list(
    map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE_PLUS_ANGLE))
transform_to_left_mount_minus = list(
    map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE_MINUS))
transform_to_left_mount_minus_angle = list(
    map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE_MINUS_ANGLE))
# transform_to_alpha_rotate = BlockWrapper(ChronoTransform, ROTATE_TO_ALPHA)

# %%
type_of_input = ChronoRevolveJoint.InputType.TORQUE

# Joints
revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z, type_of_input)

# Nodes
node_vocab = node_vocabulary.NodeVocabulary()
node_vocab.add_node(ROOT)
node_vocab.create_node("L")
node_vocab.create_node("J")
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
node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=revolve1)
node_vocab.create_node(label="J2", is_terminal=True, block_wrapper=revolve1)
node_vocab.create_node(label="J3", is_terminal=True, block_wrapper=revolve1)
node_vocab.create_node(label="J4", is_terminal=True, block_wrapper=revolve1)
node_vocab.create_node(label="J5", is_terminal=True, block_wrapper=revolve1)
node_vocab.create_node(label="J6", is_terminal=True, block_wrapper=revolve1)

node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=link[0])
node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=link[1])
node_vocab.create_node(label="L3", is_terminal=True, block_wrapper=link[2])
node_vocab.create_node(label="F1", is_terminal=True, block_wrapper=flat[0])
node_vocab.create_node(label="F2", is_terminal=True, block_wrapper=flat[1])
node_vocab.create_node(label="F3", is_terminal=True, block_wrapper=flat[2])
node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=u1)
node_vocab.create_node(label="U2", is_terminal=True, block_wrapper=u2)

node_vocab.create_node(label="TR1", is_terminal=True, block_wrapper=transform_to_right_mount[0])
node_vocab.create_node(label="TR2", is_terminal=True, block_wrapper=transform_to_right_mount[1])
node_vocab.create_node(label="TR3", is_terminal=True, block_wrapper=transform_to_right_mount[2])
node_vocab.create_node(label="TRP1",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_plus[0])
node_vocab.create_node(label="TRP2",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_plus[1])
node_vocab.create_node(label="TRP3",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_plus[2])
node_vocab.create_node(label="TRPA1",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_plus_angle[0])
node_vocab.create_node(label="TRPA2",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_plus_angle[1])
node_vocab.create_node(label="TRPA3",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_plus_angle[2])
node_vocab.create_node(label="TRM1",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_minus[0])
node_vocab.create_node(label="TRM2",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_minus[1])
node_vocab.create_node(label="TRM3",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_minus[2])
node_vocab.create_node(label="TRMA1",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_minus_angle[0])
node_vocab.create_node(label="TRMA2",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_minus_angle[1])
node_vocab.create_node(label="TRMA3",
                       is_terminal=True,
                       block_wrapper=transform_to_right_mount_minus_angle[2])

node_vocab.create_node(label="TL1", is_terminal=True, block_wrapper=transform_to_left_mount[0])
node_vocab.create_node(label="TL2", is_terminal=True, block_wrapper=transform_to_left_mount[1])
node_vocab.create_node(label="TL3", is_terminal=True, block_wrapper=transform_to_left_mount[2])
node_vocab.create_node(label="TLP1",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_plus[0])
node_vocab.create_node(label="TLP2",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_plus[1])
node_vocab.create_node(label="TLP3",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_plus[2])
node_vocab.create_node(label="TLPA1",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_plus_angle[0])
node_vocab.create_node(label="TLPA2",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_plus_angle[1])
node_vocab.create_node(label="TLPA3",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_plus_angle[2])
node_vocab.create_node(label="TLM1",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_minus[0])
node_vocab.create_node(label="TLM2",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_minus[1])
node_vocab.create_node(label="TLM3",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_minus[2])
node_vocab.create_node(label="TLMA1",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_minus_angle[0])
node_vocab.create_node(label="TLMA2",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_minus_angle[1])
node_vocab.create_node(label="TLMA3",
                       is_terminal=True,
                       block_wrapper=transform_to_left_mount_minus_angle[2])

# Defines rules
rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

rule_vocab.create_rule("InitMechanism_1", ["ROOT"], ["F", "SML", "EM"], 0, 0, [(0, 1), (1, 2)])

rule_vocab.create_rule("InitMechanism_2", ["ROOT"], ["F", "SML", "SMR", "EM", "EM"], 0, 0, [(0, 1),
                                                                                            (0, 2),
                                                                                            (1, 3),
                                                                                            (2, 4)])
rule_vocab.create_rule("InitMechanism_3_R",
                       ["ROOT"], ["F", "SML", "SMRP", "SMRM", "EM", "EM", "EM"], 0, 0, [(0, 1),
                                                                                        (0, 2),
                                                                                        (0, 3),
                                                                                        (1, 4),
                                                                                        (2, 5),
                                                                                        (3, 6)])
rule_vocab.create_rule("InitMechanism_3_R_A", ["ROOT"],
                       ["F", "SML", "SMRPA", "SMRMA", "EM", "EM", "EM"], 0, 0, [(0, 1), (0, 2),
                                                                                (0, 3), (1, 4),
                                                                                (2, 5), (3, 6)])
rule_vocab.create_rule("InitMechanism_3_L",
                       ["ROOT"], ["F", "SMLP", "SMLM", "SMR", "EM", "EM", "EM"], 0, 0, [(0, 1),
                                                                                        (0, 2),
                                                                                        (0, 3),
                                                                                        (1, 4),
                                                                                        (2, 5),
                                                                                        (3, 6)])
rule_vocab.create_rule("InitMechanism_3_L_A", ["ROOT"],
                       ["F", "SMLPA", "SMLMA", "SMR", "EM", "EM", "EM"], 0, 0, [(0, 1), (0, 2),
                                                                                (0, 3), (1, 4),
                                                                                (2, 5), (3, 6)])
rule_vocab.create_rule("InitMechanism_4", ["ROOT"],
                       ["F", "SMLP", "SMLM", "SMRP", "SMRM", "EM", "EM", "EM", "EM"], 0, 0,
                       [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8)])
rule_vocab.create_rule("InitMechanism_4_A", ["ROOT"],
                       ["F", "SMLPA", "SMLMA", "SMRPA", "SMRMA", "EM", "EM", "EM", "EM"], 0, 0,
                       [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8)])
rule_vocab.create_rule("FingerUpper", ["EM"], ["J", "L", "EM"], 0, 2, [(0, 1), (1, 2)])


rule_vocab.create_rule("TerminalJoint1", ["J"], ["J1"], 0, 0)
rule_vocab.create_rule("TerminalJoint2", ["J"], ["J2"], 0, 0)
rule_vocab.create_rule("TerminalJoint3", ["J"], ["J3"], 0, 0)
rule_vocab.create_rule("TerminalJoint4", ["J"], ["J4"], 0, 0)
rule_vocab.create_rule("TerminalJoint5", ["J"], ["J5"], 0, 0)
rule_vocab.create_rule("TerminalJoint6", ["J"], ["J6"], 0, 0)

#rule_vocab.create_rule("TerminalFlat1", ["F"], ["F1"], 0, 0)
#rule_vocab.create_rule("TerminalFlat2", ["F"], ["F2"], 0, 0)
rule_vocab.create_rule("TerminalFlat3", ["F"], ["F3"], 0, 0)

rule_vocab.create_rule("TerminalL1", ["L"], ["L1"], 0, 0)
rule_vocab.create_rule("TerminalL2", ["L"], ["L2"], 0, 0)
rule_vocab.create_rule("TerminalL3", ["L"], ["L3"], 0, 0)

rule_vocab.create_rule("TerminalTransformRight1", ["SMR"], ["TR1"], 0, 0)
rule_vocab.create_rule("TerminalTransformRight2", ["SMR"], ["TR2"], 0, 0)
rule_vocab.create_rule("TerminalTransformRight3", ["SMR"], ["TR3"], 0, 0)

rule_vocab.create_rule("TerminalTransformRightPlus1", ["SMRP"], ["TRP1"], 0, 0)
rule_vocab.create_rule("TerminalTransformRightPlus2", ["SMRP"], ["TRP2"], 0, 0)
rule_vocab.create_rule("TerminalTransformRightPlus3", ["SMRP"], ["TRP3"], 0, 0)

rule_vocab.create_rule("TerminalTransformRightPlusAngle1", ["SMRPA"], ["TRPA1"], 0, 0)
rule_vocab.create_rule("TerminalTransformRightPlusAngle2", ["SMRPA"], ["TRPA2"], 0, 0)
rule_vocab.create_rule("TerminalTransformRightPlusAngle3", ["SMRPA"], ["TRPA3"], 0, 0)

rule_vocab.create_rule("TerminalTransformRightMinus1", ["SMRM"], ["TRM1"], 0, 0)
rule_vocab.create_rule("TerminalTransformRightMinus2", ["SMRM"], ["TRM2"], 0, 0)
rule_vocab.create_rule("TerminalTransformRightMinus3", ["SMRM"], ["TRM3"], 0, 0)

rule_vocab.create_rule("TerminalTransformRightMinusAngle1", ["SMRMA"], ["TRMA1"], 0, 0)
rule_vocab.create_rule("TerminalTransformRightMinusAngle2", ["SMRMA"], ["TRMA2"], 0, 0)
rule_vocab.create_rule("TerminalTransformRightMinusAngle3", ["SMRMA"], ["TRMA3"], 0, 0)

rule_vocab.create_rule("TerminalTransformLeft1", ["SML"], ["TL1"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeft2", ["SML"], ["TL2"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeft3", ["SML"], ["TL3"], 0, 0)

rule_vocab.create_rule("TerminalTransformLeftPlus1", ["SMLP"], ["TLP1"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeftPlus2", ["SMLP"], ["TLP2"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeftPlus3", ["SMLP"], ["TLP3"], 0, 0)

rule_vocab.create_rule("TerminalTransformLeftPlusAngle1", ["SMLPA"], ["TLPA1"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeftPlusAngle2", ["SMLPA"], ["TLPA2"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeftPlusAngle3", ["SMLPA"], ["TLPA3"], 0, 0)

rule_vocab.create_rule("TerminalTransformLeftMinus1", ["SMLM"], ["TLM1"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeftMinus2", ["SMLM"], ["TLM2"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeftMinus3", ["SMLM"], ["TLM3"], 0, 0)

rule_vocab.create_rule("TerminalTransformLeftMinusAngle1", ["SMLMA"], ["TLMA1"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeftMinusAngle2", ["SMLMA"], ["TLMA2"], 0, 0)
rule_vocab.create_rule("TerminalTransformLeftMinusAngle3", ["SMLMA"], ["TLMA3"], 0, 0)

rule_vocab.create_rule("TerminalEndLimb1", ["EM"], ["U1"], 0, 0)
rule_vocab.create_rule("TerminalEndLimb2", ["EM"], ["U2"], 0, 0)

torque_dict = {
    node_vocab.get_node("J1"): 15,
    node_vocab.get_node("J2"): 10,
    node_vocab.get_node("J3"): -10,
    node_vocab.get_node("J4"): 7,
    node_vocab.get_node("J5"): 5,
    node_vocab.get_node("J6"): 3
}
time_dicst = {
    node_vocab.get_node("J1"): 0.2,
    node_vocab.get_node("J2"): 0,
    node_vocab.get_node("J3"): 0.4,
    node_vocab.get_node("J4"): 0.35,
    node_vocab.get_node("J5"): 0.1,
    node_vocab.get_node("J6"): 0.15
}
