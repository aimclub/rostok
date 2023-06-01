from itertools import product

import numpy as np
import pychrono as chrono

from rostok.block_builder_api.block_blueprints import (PrimitiveBodyBlueprint,
                                                       RevolveJointBlueprint, TransformBlueprint)
from rostok.block_builder_api.block_parameters import JointInputType
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_chrono.blocks_utils import FrameTransform
from rostok.graph_grammar import rule_vocabulary
from rostok.graph_grammar.node import ROOT
from rostok.graph_grammar.node_vocabulary import NodeVocabulary


def create_rules():

    def create_all_rules(base_name: str, in_nodes: list[str], out_nodes, in_edges, out_edges,
                         connections):
        "The function creates rules and torque dictionary for joints"

        for node in in_nodes:
            combinations = product(*out_nodes)
            for combination in combinations:
                rule_vocab.create_rule(
                    base_name + "_" + "".join([node]) + "->" + "".join(combination), [node],
                    list(combination), in_edges, out_edges, connections)

    length_link = [0.4, 0.6, 0.8]
    super_flat = PrimitiveBodyBlueprint(Box(3, 0.1, 3))
    link = list(map(lambda x: PrimitiveBodyBlueprint(Box(0.1, x, 0.3)), length_link))
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

    radial_transform = list(map(lambda x: TransformBlueprint(x), RADIAL_MOVES))
    positive_transforms = list(map(lambda x: TransformBlueprint(x), MOVES_POSITIVE))
    negative_transforms = list(map(lambda x: TransformBlueprint(x), MOVES_NEGATIVE))
    reverse_transform = TransformBlueprint(REVERSE_Y)
    turn_transform_P = TransformBlueprint(TURN_P)
    turn_transform_N = TransformBlueprint(TURN_N)

    revolve = RevolveJointBlueprint(JointInputType.TORQUE)
    # Nodes
    node_vocab = NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node(label="F")
    node_vocab.create_node(label="RF")
    node_vocab.create_node(label="PF")
    node_vocab.create_node(label="NF")
    node_vocab.create_node(label="RPF")
    node_vocab.create_node(label="RNF")
    node_vocab.create_node(label="FT", is_terminal=True, block_blueprint=super_flat)
    node_vocab.create_node(label="RE", is_terminal=True, block_blueprint=reverse_transform)
    node_vocab.create_node(label="RT")
    node_vocab.create_node(label="RT1", is_terminal=True, block_blueprint=radial_transform[0])
    node_vocab.create_node(label="RT2", is_terminal=True, block_blueprint=radial_transform[1])
    node_vocab.create_node(label="RT3", is_terminal=True, block_blueprint=radial_transform[2])
    node_vocab.create_node(label="FG")
    node_vocab.create_node(label="J1", is_terminal=True, block_blueprint=revolve)
    node_vocab.create_node(label="L")
    node_vocab.create_node(label="L1", is_terminal=True, block_blueprint=link[0])
    node_vocab.create_node(label="L2", is_terminal=True, block_blueprint=link[1])
    node_vocab.create_node(label="L3", is_terminal=True, block_blueprint=link[2])
    node_vocab.create_node(label="TP")
    node_vocab.create_node(label="TP1", is_terminal=True, block_blueprint=positive_transforms[0])
    node_vocab.create_node(label="TP2", is_terminal=True, block_blueprint=positive_transforms[1])
    node_vocab.create_node(label="TP3", is_terminal=True, block_blueprint=positive_transforms[2])
    node_vocab.create_node(label="TN")
    node_vocab.create_node(label="TN1", is_terminal=True, block_blueprint=negative_transforms[0])
    node_vocab.create_node(label="TN2", is_terminal=True, block_blueprint=negative_transforms[1])
    node_vocab.create_node(label="TN3", is_terminal=True, block_blueprint=negative_transforms[2])
    node_vocab.create_node(label="TURN_P", is_terminal=True, block_blueprint=turn_transform_P)
    node_vocab.create_node(label="TURN_N", is_terminal=True, block_blueprint=turn_transform_N)

    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)
    rule_vocab.create_rule("Init", ["ROOT"], ["FT", "F", "RF", "PF", "NF", "RPF", "RNF"], 0, 0,
                           [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])
    create_all_rules("AddFinger", ["F"], [["RT1", "RT2", "RT3"],  ["FG"]], 0, 0, [(0, 1)])
    rule_vocab.create_rule("RemoveFinger", ["F"], [], 0, 0, [])

    create_all_rules("AddFinger_R", ["RF"], [["RE"], ["RT1", "RT2", "RT3"], ["FG"]], 0, 0,
                     [(0, 1), (1, 2)])
    rule_vocab.create_rule("RemoveFinger_R", ["RF"], [], 0, 0, [])

    create_all_rules("Phalanx", ["FG"], [["J1"], ["L1", "L2", "L3"], ["FG"]], 0, 0, [(0, 1),
                                                                                     (1, 2)])

    rule_vocab.create_rule("Remove_FG", ["FG"], [], 0, 0, [])

    create_all_rules("AddFinger_P", ["PF"],
                     [["RT1", "RT2", "RT3"], ["TP1", "TP2", "TP3"], ["FG"]], 0, 0, [(0, 1),
                                                                                            (1, 2)])
    create_all_rules("AddFinger_PT", ["PF"], [["TURN_N"], ["RT1", "RT2", "RT3"], ["FG"]], 0,
                     0, [(0, 1), (1, 2)])
    rule_vocab.create_rule("RemoveFinger_P", ["PF"], [], 0, 0, [])

    create_all_rules("AddFinger_N", ["NF"],
                     [["RT1", "RT2", "RT3"], ["TN1", "TN2", "TN3"], ["FG"]], 0, 0, [(0, 1),
                                                                                            (1, 2)])
    create_all_rules("AddFinger_NT", ["NF"], [["TURN_P"], ["RT1", "RT2", "RT3"], ["FG"]], 0,
                     0, [(0, 1), (1, 2)])
    rule_vocab.create_rule("RemoveFinger_N", ["NF"], [], 0, 0, [])

    create_all_rules("AddFinger_RP", ["RPF"],
                     [["RE"], ["RT1", "RT2", "RT3"], ["TP1", "TP2", "TP3"], ["FG"]], 0, 0,
                     [(0, 1), (1, 2), (2, 3)])
    create_all_rules("AddFinger_RPT", ["RPF"],
                     [["RE"], ["TURN_N"], ["RT1", "RT2", "RT3"], ["FG"]], 0, 0, [(0, 1),
                                                                                         (1, 2),
                                                                                         (2, 3)])
    rule_vocab.create_rule("RemoveFinger_RP", ["RPF"], [], 0, 0, [])

    create_all_rules("AddFinger_RN", ["RNF"],
                     [["RE"], ["RT1", "RT2", "RT3"], ["TN1", "TN2", "TN3"], ["FG"]], 0, 0,
                     [(0, 1), (1, 2), (2, 3)])
    create_all_rules("AddFinger_RNT", ["RNF"],
                     [["RE"], ["TURN_P"], ["RT1", "RT2", "RT3"],  ["FG"]], 0, 0, [(0, 1),
                                                                                         (1, 2),
                                                                                         (2, 3)])
    rule_vocab.create_rule("RemoveFinger_RN", ["RNF"], [], 0, 0, [])

    return rule_vocab


if __name__ == "__main__":
    rv = create_rules()
    print(rv)
