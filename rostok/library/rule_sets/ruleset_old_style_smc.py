import numpy as np
import pychrono as chrono

from rostok.block_builder_chrono.blocks_utils import FrameTransform
from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.graph_grammar.node import ROOT
from rostok.graph_grammar import rule_vocabulary
from rostok.block_builder_api.block_blueprints import TransformBlueprint, PrimitiveBodyBlueprint, RevolveJointBlueprint
from rostok.utils.dataset_materials.material_dataclass_manipulating import DefaultChronoMaterialSMC
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_api.block_parameters import JointInputType


def create_rules(tendon = True):

    length_link = [0.06, 0.08, 0.1]
    super_flat = PrimitiveBodyBlueprint(Box(0.3, 0.01, 0.30), material=DefaultChronoMaterialSMC(), color=[255,0,0])
    base = PrimitiveBodyBlueprint(Box(0.03, 0.03, 0.03), material=DefaultChronoMaterialSMC(), color=[0,120, 255])
    link = list(map(lambda x: PrimitiveBodyBlueprint(Box(0.03, x, 0.03), material=DefaultChronoMaterialSMC(), color= [0, 120, 255]), length_link))
    radial_move_values = [0.09, 0.105, 0.12]
    #radial_move_values = [0.65, 0.85, 1.05 ]
    RADIAL_MOVES = list(map(lambda x: FrameTransform([x, 0, 0], [1, 0, 0, 0]), radial_move_values))
    tan_move_values = [0.04, 0.06, 0.08]
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
    radial_transform = list(map(lambda x: TransformBlueprint(x), RADIAL_MOVES))
    positive_transforms = list(map(lambda x: TransformBlueprint(x), MOVES_POSITIVE))
    negative_transforms = list(map(lambda x: TransformBlueprint(x), MOVES_NEGATIVE))
    reverse_transform = TransformBlueprint(REVERSE_Y)
    turn_transform_P = TransformBlueprint(TURN_P)
    turn_transform_N = TransformBlueprint(TURN_N)

    #revolve = RevolveJointBlueprint(JointInputType.POSITION)
    revolve = RevolveJointBlueprint(JointInputType.TORQUE, material=DefaultChronoMaterialSMC(), stiffness=0.02 ,damping=0)
    revolve_45 = RevolveJointBlueprint(JointInputType.TORQUE, starting_angle=45)
    no_control = RevolveJointBlueprint(JointInputType.UNCONTROL, stiffness=0.02 ,damping=0)
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
    node_vocab.create_node(label="FG1")
    if tendon:
        node_vocab.create_node(label="J", is_terminal=True, block_blueprint=no_control)
    else:
        node_vocab.create_node(label="J", is_terminal=True, block_blueprint=revolve)
    node_vocab.create_node(label="L")
    node_vocab.create_node(label="B", is_terminal=True, block_blueprint=base)
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
    rule_vocab.create_rule("Init", ["ROOT"], ["FT", "F", "RF", "PF", "NF", "RPF", "RNF"], 0, 0, [(0,1), (0,2), (0,3), (0,4), (0,5),(0, 6)])
    rule_vocab.create_rule("AddFinger", ["F"], [ "RT", "FG1"], 0, 0, [(0, 1)])
    rule_vocab.create_rule("RemoveFinger", ["F"], [], 0, 0, [])
    rule_vocab.create_rule("AddFinger_R", ["RF"], ["RE", "RT", "FG1"], 0, 0, [(0, 1), (1, 2)])
    rule_vocab.create_rule("RemoveFinger_R", ["RF"], [], 0, 0, [])
    rule_vocab.create_rule("Terminal_Radial_Translate1", ["RT"], ["RT1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Radial_Translate2", ["RT"], ["RT2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Radial_Translate3", ["RT"], ["RT3"], 0, 0, [])

    rule_vocab.create_rule("Phalanx", ["FG"], ["J", "L", "FG"], 0, 0, [(0, 1), (1, 2)])
    rule_vocab.create_rule("Phalanx_1", ["FG1"], ["B","J", "L", "FG"], 0, 0, [(0, 1), (1, 2),(2,3)])
    rule_vocab.create_rule("Terminal_Link1", ["L"], ["L1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Link2", ["L"], ["L2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Link3", ["L"], ["L3"], 0, 0, [])
    rule_vocab.create_rule("Remove_FG", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("Remove_FG1", ["FG1"], [], 0, 0, [])

    rule_vocab.create_rule("AddFinger_P", ["PF"], ["RT", "TP", "FG1"], 0, 0, [(0, 1), (1, 2)])
    rule_vocab.create_rule("AddFinger_PT", ["PF"], ["TURN_N", "RT",  "FG1"], 0, 0, [(0, 1), (1, 2)])
    rule_vocab.create_rule("RemoveFinger_P", ["PF"], [], 0, 0, [])
    rule_vocab.create_rule("AddFinger_N", ["NF"], ["RT", "TN", "FG"], 0, 0, [(0, 1), (1, 2)])
    rule_vocab.create_rule("AddFinger_NT", ["NF"], ["TURN_P", "RT", "FG"], 0, 0, [(0, 1), (1, 2)])
    rule_vocab.create_rule("RemoveFinger_N", ["NF"], [], 0, 0, [])
    rule_vocab.create_rule("AddFinger_RP", ["RPF"], ["RE", "RT", "TP", "FG"], 0, 0, [(0, 1), (1, 2), (2, 3)])
    rule_vocab.create_rule("AddFinger_RPT", ["RPF"], ["RE", "TURN_N", "RT", "FG"], 0, 0, [(0, 1), (1, 2),(2, 3)])
    rule_vocab.create_rule("RemoveFinger_RP", ["RPF"], [], 0, 0, [])
    rule_vocab.create_rule("AddFinger_RN", ["RNF"], ["RE", "RT", "TN", "FG"], 0, 0, [(0, 1), (1, 2),(2, 3)])
    rule_vocab.create_rule("AddFinger_RNT", ["RNF"], ["RE", "TURN_P", "RT", "FG"], 0, 0, [(0, 1), (1, 2), (2, 3)])
    rule_vocab.create_rule("RemoveFinger_RN", ["RNF"], [], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Translate1", ["TP"], ["TP1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Translate2", ["TP"], ["TP2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Translate3", ["TP"], ["TP3"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Translate1", ["TN"], ["TN1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Translate2", ["TN"], ["TN2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Translate3", ["TN"], ["TN3"], 0, 0, [])
    return rule_vocab


# if __name__ == "__main__":
#     rv, _ =create_rules()
#     print(rv)
#     print(rv.get_rule("Failed_Path").is_terminal)
