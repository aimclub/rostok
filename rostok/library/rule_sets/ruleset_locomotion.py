import numpy as np
import pychrono as chrono

from rostok.block_builder_chrono_alt.blocks_utils import FrameTransform
from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.graph_grammar.node import ROOT
from rostok.graph_grammar import rule_vocabulary
from rostok.block_builder_api.block_blueprints import TransformBlueprint, PrimitiveBodyBlueprint, RevolveJointBlueprint
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_api.block_parameters import JointInputType


def create_rules():

    length_link = [0.4, 0.6, 0.8]
    main_body = PrimitiveBodyBlueprint(Box(0.8, 0.8, 0.8))
    foot = PrimitiveBodyBlueprint(Box(0.2, 0.2, 0.5))
    link = list(map(lambda x: PrimitiveBodyBlueprint(Box(0.1, x, 0.1)), length_link))
    Z_MOVE_POSITIVE = FrameTransform([0, 0, 0.7], [1, 0, 0, 0])
    Z_MOVE_NEGATIVE = FrameTransform([0, 0, -0.7], [1, 0, 0, 0])
    Y_MOVE_NEGATIVE = FrameTransform([0, -0.4, 0], [1, 0, 0, 0])

    z_move_positive = TransformBlueprint(Z_MOVE_POSITIVE)
    z_move_negative = TransformBlueprint(Z_MOVE_NEGATIVE)
    y_move_negative = TransformBlueprint(Y_MOVE_NEGATIVE)

    def rotation_y(alpha):
        quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
        return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]

    turn_90_transform = TransformBlueprint(FrameTransform([0, 0, 0], rotation_y(90)))
    turn_m90_transform = TransformBlueprint(FrameTransform([0, 0, 0], rotation_y(-90)))
    revolve = RevolveJointBlueprint(JointInputType.TORQUE, stiffness=100, damping=0.2)
    # Nodes
    node_vocab = NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node(label="MB", is_terminal=True, block_blueprint=main_body)
    node_vocab.create_node(label="ZTP", is_terminal=True, block_blueprint=z_move_positive)
    node_vocab.create_node(label="ZTN", is_terminal=True, block_blueprint=z_move_negative)
    node_vocab.create_node(label="YTN", is_terminal=True, block_blueprint=y_move_negative)
    node_vocab.create_node(label="FG")
    node_vocab.create_node(label="J")
    node_vocab.create_node(label="JT", is_terminal=True, block_blueprint=turn_90_transform)
    node_vocab.create_node(label="JTN", is_terminal=True, block_blueprint=turn_m90_transform)
    node_vocab.create_node(label="J1", is_terminal=True, block_blueprint=revolve)
    node_vocab.create_node(label="F", is_terminal=True, block_blueprint=foot)

    node_vocab.create_node(label="L")
    node_vocab.create_node(label="L1", is_terminal=True, block_blueprint=link[0])
    node_vocab.create_node(label="L2", is_terminal=True, block_blueprint=link[1])
    node_vocab.create_node(label="L3", is_terminal=True, block_blueprint=link[2])

    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)
    rule_vocab.create_rule("Init", ["ROOT"], ["MB", "ZTP", "YTN", "J1", "JT", "FG"], 0, 0, [(0, 1),
                                                                                            (1, 2),
                                                                                            (2, 3),
                                                                                            (3, 4),
                                                                                            (4, 5)])
    rule_vocab.create_rule("Init_2", ["MB"], ["MB", "ZTN", "YTN", "J1", "JT", "FG"], 0, 0, [(0, 1),
                                                                                            (1, 2),
                                                                                            (2, 3),
                                                                                            (3, 4),
                                                                                            (4, 5)])
    rule_vocab.create_rule("Init_3", ["ROOT"], ["MB", "ZTP", "YTN", "FG"], 0, 0, [(0, 1),
                                                                                            (1, 2),
                                                                                            (2, 3)])
    rule_vocab.create_rule("Init_4", ["MB"], ["MB", "ZTN", "YTN",  "FG"], 0, 0, [(0, 1),
                                                                                            (1, 2),
                                                                                            (2, 3)])
    
    rule_vocab.create_rule("Init_5", ["ROOT"], ["MB", "ZTP", "YTN","J1" ,"FG"], 0, 0, [(0, 1),
                                                                                            (1, 2),
                                                                                            (2, 3),(3,4)])
    rule_vocab.create_rule("Init_6", ["MB"], ["MB", "ZTN", "YTN","J1",  "FG"], 0, 0, [(0, 1),
                                                                                            (1, 2),
                                                                                            (2, 3),(3,4)])

    rule_vocab.create_rule("Terminal_Link1", ["L"], ["L1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Link2", ["L"], ["L2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Link3", ["L"], ["L3"], 0, 0, [])
    rule_vocab.create_rule("Phalanx", ["FG"], ["J1", "L", "FG"], 0, 0, [(0, 1), (1, 2)])
    rule_vocab.create_rule("Double_Joint_Phalanx", ["FG"], ["J1", "JT", "J1", "L", "FG"], 0, 0,
                           [(0, 1), (1, 2), (2, 3), (3, 4)])
    rule_vocab.create_rule("Remove_FG", ["FG"], ["J1", "JT", "J1","F"], 0, 0, [(0, 1), (1, 2), (2, 3)])
    rule_vocab.create_rule("Remove_FG_2", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Rev_Turn", ["FG"], ["JTN", "FG"], 0, 1,[(0,1)])
    rule_vocab.create_rule("Terminal_Joint1", ["J"], ["J1"], 0, 0, [])

    return rule_vocab


from rostok.graph_grammar.node import GraphGrammar


def get_bip():
    G = GraphGrammar()
    rules = [
        "Init", "Init_2", "Phalanx", "Phalanx","Rev_Turn","Rev_Turn" ,"Phalanx", "Phalanx", "Remove_FG", "Remove_FG",
        "Terminal_Link2", "Terminal_Link2", "Terminal_Link3", "Terminal_Link3"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G

def get_bip_single():
    G = GraphGrammar()
    rules = [
        "Init_3", "Init_4", "Phalanx", "Phalanx", "Phalanx", "Phalanx", "Remove_FG_2", "Remove_FG_2",
        "Terminal_Link2", "Terminal_Link2", "Terminal_Link3", "Terminal_Link3"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G


def get_box():
    G = GraphGrammar()
    rules = [
        "Init_3", "Init_4",  "Remove_FG_2", "Remove_FG_2"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G

def get_box_joints():
    G = GraphGrammar()
    rules = [
        "Init_5", "Init_6",  "Remove_FG_2", "Remove_FG_2"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G

def get_box_one_joint():
    G = GraphGrammar()
    rules = [
        "Init_5", "Remove_FG_2"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        G.apply_rule(rule_vocabul.get_rule(rule))

    return G