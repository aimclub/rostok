from itertools import product

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


def create_rules():

    def create_all_rules(base_name: str, in_nodes: list[str], out_nodes, in_edges, out_edges,
                         connections):
        for node in in_nodes:
            combinations = product(*out_nodes)
            for combination in combinations:
                rule_vocab.create_rule(
                    base_name + "_" + "".join([node]) + "->" + "".join(combination), [node],
                    list(combination), in_edges, out_edges, connections)

    length_link = [0.4, 0.6, 0.8]
    super_flat = BlockWrapper(UniversalBox, x=3, y=0.1, z=3)
    link = list(map(lambda x: BlockWrapper(UniversalBox, x=0.1, y=x, z=0.3), length_link))
    radial_move_values = [0.9, 1.05, 1.2]
    RADIAL_MOVES = list(map(lambda x: FrameTransform([x, 0, 0], [1, 0, 0, 0]), radial_move_values))
    tan_move_values = [0.4, 0.6, 0.8]
    MOVES_POSITIVE = list(map(lambda x: FrameTransform([0, 0, x], [1, 0, 0, 0]), tan_move_values))
    MOVES_NEGATIVE = list(map(lambda x: FrameTransform([0, 0, -x], [1, 0, 0, 0]), tan_move_values))

    def rotation_y(alpha):
        quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
        return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]

    def rotation_z(alpha):
        quat_Z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
        return [quat_Z_ang_alpha.e0, quat_Z_ang_alpha.e1, quat_Z_ang_alpha.e2, quat_Z_ang_alpha.e3]

    REVERSE_Y = FrameTransform([0, 0, 0], [0, 0, 1, 0])
    turn_const = 60
    TURN_P = FrameTransform([0, 0, 0], rotation_y(turn_const))
    TURN_N = FrameTransform([0, 0, 0], rotation_y(-turn_const))
    # MOVE = FrameTransform([1, 0, 0], rotation(45))
    radial_transform = list(map(lambda x: BlockWrapper(ChronoTransform, x), RADIAL_MOVES))
    positive_transforms = list(map(lambda x: BlockWrapper(ChronoTransform, x), MOVES_POSITIVE))
    negative_transforms = list(map(lambda x: BlockWrapper(ChronoTransform, x), MOVES_NEGATIVE))
    reverse_transform = BlockWrapper(ChronoTransform, REVERSE_Y)
    turn_transform_P = BlockWrapper(ChronoTransform, TURN_P)
    turn_transform_N = BlockWrapper(ChronoTransform, TURN_N)

    type_of_input = ChronoRevolveJoint.InputType.TORQUE
    revolve = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z, type_of_input)
    # Nodes
    node_vocab = node_vocabulary.NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node(label="F")
    node_vocab.create_node(label="RF")
    node_vocab.create_node(label="PF")
    node_vocab.create_node(label="NF")
    node_vocab.create_node(label="RPF")
    node_vocab.create_node(label="RNF")
    node_vocab.create_node(label="FT", is_terminal=True, block_wrapper=super_flat)
    node_vocab.create_node(label="RE", is_terminal=True, block_wrapper=reverse_transform)
    node_vocab.create_node(label="RT")
    node_vocab.create_node(label="RT1", is_terminal=True, block_wrapper=radial_transform[0])
    node_vocab.create_node(label="RT2", is_terminal=True, block_wrapper=radial_transform[1])
    node_vocab.create_node(label="RT3", is_terminal=True, block_wrapper=radial_transform[2])
    node_vocab.create_node(label="FG")
    node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="J2", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="J3", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="J4", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="J5", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="J6", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="J7", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="J8", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="J9", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="J10", is_terminal=True, block_wrapper=revolve)
    node_vocab.create_node(label="L")
    node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=link[0])
    node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=link[1])
    node_vocab.create_node(label="L3", is_terminal=True, block_wrapper=link[2])
    node_vocab.create_node(label="TP")
    node_vocab.create_node(label="TP1", is_terminal=True, block_wrapper=positive_transforms[0])
    node_vocab.create_node(label="TP2", is_terminal=True, block_wrapper=positive_transforms[1])
    node_vocab.create_node(label="TP3", is_terminal=True, block_wrapper=positive_transforms[2])
    node_vocab.create_node(label="TN")
    node_vocab.create_node(label="TN1", is_terminal=True, block_wrapper=negative_transforms[0])
    node_vocab.create_node(label="TN2", is_terminal=True, block_wrapper=negative_transforms[1])
    node_vocab.create_node(label="TN3", is_terminal=True, block_wrapper=negative_transforms[2])
    node_vocab.create_node(label="RP", is_terminal=True, block_wrapper=turn_transform_P)
    node_vocab.create_node(label="RN", is_terminal=True, block_wrapper=turn_transform_N)

    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)
    rule_vocab.create_rule("Init", ["ROOT"], ["FT", "F", "RF", "PF", "NF", "RPF", "RNF"], 0, 0,
                           [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])
    create_all_rules("AddFinger", ["F"], [["RT1", "RT2", "RT3"], ["RE"], ["FG"]], 0, 0, [(0, 1),
                                                                                         (1, 2)])
    rule_vocab.create_rule("RemoveFinger", ["F"], [], 0, 0, [])

    create_all_rules("AddFinger_R", ["RF"], [["RE"], ["RT1", "RT2", "RT3"], ["RE"], ["FG"]], 0, 0,
                     [(0, 1), (1, 2), (2, 3)])
    rule_vocab.create_rule("RemoveFinger_R", ["RF"], [], 0, 0, [])

    create_all_rules("Phalanx", ["FG"], [["J1","J2","J3","J4","J5","J6","J7","J8","J9", "J10"], ["L1", "L2", "L3"], ["FG"]], 0, 0, [(0, 1),
                                                                                     (1, 2)])

    rule_vocab.create_rule("Remove_FG_1", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG_2", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG_3", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG_4", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG_5", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG_6", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG_7", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG_8", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG_9", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG_10", ["FG"], [], 0, 0, [])

    create_all_rules("AddFinger_P", ["PF"],
                     [["RT1", "RT2", "RT3"], ["TP1", "TP2", "TP3"], ["RE"], ["FG"]], 0, 0, [(0, 1),
                                                                                            (1, 2),
                                                                                            (2, 3)])
    create_all_rules("AddFinger_PT", ["PF"], [["RN"], ["RT1", "RT2", "RT3"], ["RE"], ["FG"]], 0,
                     0, [(0, 1), (1, 2), (2, 3)])
    rule_vocab.create_rule("RemoveFinger_P", ["PF"], [], 0, 0, [])

    create_all_rules("AddFinger_N", ["NF"],
                     [["RT1", "RT2", "RT3"], ["TN1", "TN2", "TN3"], ["RE"], ["FG"]], 0, 0, [(0, 1),
                                                                                            (1, 2),
                                                                                            (2, 3)])
    create_all_rules("AddFinger_NT", ["NF"], [["RP"], ["RT1", "RT2", "RT3"], ["RE"], ["FG"]], 0,
                     0, [(0, 1), (1, 2), (2, 3)])
    rule_vocab.create_rule("RemoveFinger_N", ["NF"], [], 0, 0, [])

    create_all_rules("AddFinger_RP", ["RPF"],
                     [["RE"], ["RT1", "RT2", "RT3"], ["TP1", "TP2", "TP3"], ["RE"], ["FG"]], 0, 0,
                     [(0, 1), (1, 2), (2, 3), (3, 4)])
    create_all_rules("AddFinger_RPT", ["RPF"],
                     [["RE"], ["RN"], ["RT1", "RT2", "RT3"], ["RE"], ["FG"]], 0, 0, [(0, 1),
                                                                                         (1, 2),
                                                                                         (2, 3),
                                                                                         (3, 4)])
    rule_vocab.create_rule("RemoveFinger_RP", ["RPF"], [], 0, 0, [])

    create_all_rules("AddFinger_RN", ["RNF"],
                     [["RE"], ["RT1", "RT2", "RT3"], ["TN1", "TN2", "TN3"], ["RE"], ["FG"]], 0, 0,
                     [(0, 1), (1, 2), (2, 3), (3, 4)])
    create_all_rules("AddFinger_RNT", ["RNF"],
                     [["RE"], ["RP"], ["RT1", "RT2", "RT3"], ["RE"], ["FG"]], 0, 0, [(0, 1),
                                                                                         (1, 2),
                                                                                         (2, 3),
                                                                                         (3, 4)])
    rule_vocab.create_rule("RemoveFinger_RN", ["RNF"], [], 0, 0, [])
    torque_dict = {
        node_vocab.get_node("J1"): 6,
        node_vocab.get_node("J2"): 7,
        node_vocab.get_node("J3"): 8,
        node_vocab.get_node("J4"): 9,
        node_vocab.get_node("J5"): 10,
        node_vocab.get_node("J6"): 11,
        node_vocab.get_node("J7"): 12,
        node_vocab.get_node("J8"): 13,
        node_vocab.get_node("J9"): 14,
        node_vocab.get_node("J10"): 15
    }
    return rule_vocab, torque_dict


if __name__ == "__main__":
    rv = create_rules()
    print(rv)
