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
from rostok.utils.dataset_materials.material_dataclass_manipulating import \
    DefaultChronoMaterialSMC


def get_density_box(mass: float, box: Box):
    volume = box.height_z * box.length_y * box.width_x
    return mass / volume


def create_rules(tendon=True):

    length_link = [0.05, 0.06, 0.075]
    super_flat = PrimitiveBodyBlueprint(Box(0.3, 0.01, 0.30),
                                        #material=DefaultChronoMaterialSMC(),
                                        color=[255, 0, 0])
    base = PrimitiveBodyBlueprint(Box(0.03, 0.01, 0.03),
                                  #material=DefaultChronoMaterialSMC(),
                                  color=[0, 120, 255], density= 10000)
    #sets effective density for the
    link_mass = (28 + 1.62 + 2.77) * 1e-3
    link = list(
        map(
            lambda x: PrimitiveBodyBlueprint(Box(0.035, x, 0.035),
                                             #material=DefaultChronoMaterialSMC(),
                                             color=[0, 120, 255],
                                             density=get_density_box(link_mass, Box(0.035, x, 0.035))
                                            ), length_link))

    radial_move_values = [0.06, 0.085, 0.11]

    RADIAL_MOVES = list(map(lambda x: FrameTransform([x, 0, 0], [1, 0, 0, 0]), radial_move_values))
    tan_move_values = [0.04, 0.065, 0.09]
    MOVES_POSITIVE = list(map(lambda x: FrameTransform([0, 0, x], [1, 0, 0, 0]), tan_move_values))
    MOVES_NEGATIVE = list(map(lambda x: FrameTransform([0, 0, -x], [1, 0, 0, 0]), tan_move_values))

    def rotation_y(alpha):
        quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
        return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]

    def rotation_z(alpha):
        quat_Z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
        return [quat_Z_ang_alpha.e0, quat_Z_ang_alpha.e1, quat_Z_ang_alpha.e2, quat_Z_ang_alpha.e3]

    REVERSE_Y = FrameTransform([0, 0, 0], [0, 0, 1, 0])
    turn_const_0 = 0
    turn_const_1 = 30
    turn_const_2 = 60
    TURN_P_0 = FrameTransform([0, 0, 0], rotation_y(turn_const_0))
    TURN_N_0 = FrameTransform([0, 0, 0], rotation_y(-turn_const_0))
    TURN_P_1 = FrameTransform([0, 0, 0], rotation_y(turn_const_1))
    TURN_N_1 = FrameTransform([0, 0, 0], rotation_y(-turn_const_1))
    TURN_P_2 = FrameTransform([0, 0, 0], rotation_y(turn_const_2))
    TURN_N_2 = FrameTransform([0, 0, 0], rotation_y(-turn_const_2))

    radial_transform = list(map(lambda x: TransformBlueprint(x), RADIAL_MOVES))
    positive_transforms = list(map(lambda x: TransformBlueprint(x), MOVES_POSITIVE))
    negative_transforms = list(map(lambda x: TransformBlueprint(x), MOVES_NEGATIVE))
    reverse_transform = TransformBlueprint(REVERSE_Y)
    turn_transform_P_0 = TransformBlueprint(TURN_P_0)
    turn_transform_N_0 = TransformBlueprint(TURN_N_0)
    turn_transform_P_1 = TransformBlueprint(TURN_P_1)
    turn_transform_N_1 = TransformBlueprint(TURN_N_1)
    turn_transform_P_2 = TransformBlueprint(TURN_P_2)
    turn_transform_N_2 = TransformBlueprint(TURN_N_2)

    #revolve = RevolveJointBlueprint(JointInputType.POSITION)
    revolve = RevolveJointBlueprint(JointInputType.TORQUE,
                                    stiffness=0.02,
                                    damping=0)
    mass_joint = (10/3 + 0.51*2 + 0.64 + 1.3) * 1e-3  #0.012
    joint_radius_base = 0.015
    joint_radius = 0.015
    joint_length = 0.03
    density_joint = (mass_joint / (0.03 * 3.14 * joint_radius**2))

    stiffness = [0.095, 0.07] 
    preload = [0.8, 0.8]
    no_control = list(map(lambda x, y: RevolveJointBlueprint(JointInputType.UNCONTROL,
                                       stiffness=x,
                                       damping=0.01,
                                       offset=0.0085,
                                       radius=joint_radius,
                                       length=joint_length,
                                       density=density_joint,
                                       equilibrium_position = y), stiffness, preload))

    stiffness_base = [0.17, 0.095, 0.07]
    preload_base = [0, 0, 0]
    no_control_base = list(map(lambda x, y: RevolveJointBlueprint(JointInputType.UNCONTROL,
                                            stiffness=0.01,
                                            damping=0.01,
                                            offset=0,
                                            radius=joint_radius_base,
                                            length=joint_length,
                                            density=density_joint,
                                            equilibrium_position = 0), stiffness_base, preload_base))
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
        node_vocab.create_node(label="J")
        node_vocab.create_node(label="J_1", is_terminal=True, block_blueprint=no_control[0])
        node_vocab.create_node(label="J_2", is_terminal=True, block_blueprint=no_control[1])
        

        node_vocab.create_node(label="JB")
        node_vocab.create_node(label="JB_1", is_terminal=True, block_blueprint=no_control_base[0])
        node_vocab.create_node(label="JB_2", is_terminal=True, block_blueprint=no_control_base[1])
        node_vocab.create_node(label="JB_3", is_terminal=True, block_blueprint=no_control_base[2])
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
    
    node_vocab.create_node(label="TURN_P")
    node_vocab.create_node(label="TURN_N")

    node_vocab.create_node(label="TURN_P_0", is_terminal=True, block_blueprint=turn_transform_P_0)
    node_vocab.create_node(label="TURN_N_0", is_terminal=True, block_blueprint=turn_transform_N_0)
    node_vocab.create_node(label="TURN_P_1", is_terminal=True, block_blueprint=turn_transform_P_1)
    node_vocab.create_node(label="TURN_N_1", is_terminal=True, block_blueprint=turn_transform_N_1)
    node_vocab.create_node(label="TURN_P_2", is_terminal=True, block_blueprint=turn_transform_P_2)
    node_vocab.create_node(label="TURN_N_2", is_terminal=True, block_blueprint=turn_transform_N_2)


    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)
    rule_vocab.create_rule("Init", ["ROOT"], ["FT", "F", "RF", "PF", "NF", "RPF", "RNF"], 0, 0,
                           [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])
    rule_vocab.create_rule("AddFinger", ["F"], ["RT", "B", "JB", "L", "FG"], 0, 0, [(0, 1), (1,2), (2,3), (3, 4)])
    rule_vocab.create_rule("RemoveFinger", ["F"], [], 0, 0, [])
    
    rule_vocab.create_rule("AddFinger_R", ["RF"], ["RE", "RT", "B", "JB", "L", "FG"], 0, 0, [(0, 1), (1, 2), (2,3), (3,4), (4,5)])
    rule_vocab.create_rule("RemoveFinger_R", ["RF"], [], 0, 0, [])

    rule_vocab.create_rule("Terminal_Radial_Translate1", ["RT"], ["RT1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Radial_Translate2", ["RT"], ["RT2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Radial_Translate3", ["RT"], ["RT3"], 0, 0, [])

    rule_vocab.create_rule("Phalanx", ["FG"], ["J", "L", "FG"], 0, 0, [(0, 1), (1, 2)])
    # rule_vocab.create_rule("Phalanx_1", ["FG1"], ["B", "JB", "L", "FG"], 0, 0, [(0, 1), (1, 2),
    #                                                                       (2, 3)])
    rule_vocab.create_rule('Terminal_Joint_1', ['J'], ["J_1"], 0, 0, [])
    rule_vocab.create_rule('Terminal_Joint_2', ['J'], ["J_2"], 0, 0, [])

    rule_vocab.create_rule('Terminal_Base_Joint_1', ['JB'], ["JB_1"], 0, 0, [])
    rule_vocab.create_rule('Terminal_Base_Joint_2', ['JB'], ["JB_2"], 0, 0, [])
    rule_vocab.create_rule('Terminal_Base_Joint_3', ['JB'], ["JB_3"], 0, 0, [])

    rule_vocab.create_rule("Terminal_Link1", ["L"], ["L1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Link2", ["L"], ["L2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Link3", ["L"], ["L3"], 0, 0, [])
    rule_vocab.create_rule("Remove_FG", ["FG"], [], 0, 0, [])
    # rule_vocab.create_rule("Remove_FG1", ["FG1"], [], 0, 0, [])

    rule_vocab.create_rule("AddFinger_P", ["PF"], ["RT", "TP", "TURN_N", "B", "JB", "L", "FG"], 0, 0, [(0, 1), (1, 2), (2,3), (3,4), (4,5), (5,6)])
    rule_vocab.create_rule("RemoveFinger_P", ["PF"], [], 0, 0, [])

    rule_vocab.create_rule("AddFinger_N", ["NF"], ["RT", "TN", "TURN_P" , "B", "JB", "L", "FG"], 0, 0, [(0, 1), (1, 2), (2,3), (3,4), (4,5), (5,6)])
    rule_vocab.create_rule("RemoveFinger_N", ["NF"], [], 0, 0, [])

    rule_vocab.create_rule("AddFinger_RP", ["RPF"], ["RE", "RT", "TP", "TURN_N", "B", "JB", "L", "FG"], 0, 0,
                           [(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6,7)])
    rule_vocab.create_rule("RemoveFinger_RP", ["RPF"], [], 0, 0, [])
    rule_vocab.create_rule("AddFinger_RN", ["RNF"], ["RE", "RT", "TN", "TURN_P", "B", "JB", "L", "FG"], 0, 0,
                           [(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6,7)])
    rule_vocab.create_rule("RemoveFinger_RN", ["RNF"], [], 0, 0, [])

    rule_vocab.create_rule("Terminal_Positive_Translate1", ["TP"], ["TP1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Translate2", ["TP"], ["TP2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Translate3", ["TP"], ["TP3"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Translate1", ["TN"], ["TN1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Translate2", ["TN"], ["TN2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Translate3", ["TN"], ["TN3"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Turn_0", ["TURN_P"], ["TURN_P_0"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Turn_1", ["TURN_P"], ["TURN_P_1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Turn_2", ["TURN_P"], ["TURN_P_2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Turn_0", ["TURN_N"], ["TURN_N_0"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Turn_1", ["TURN_N"], ["TURN_N_1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Turn_2", ["TURN_N"], ["TURN_N_2"], 0, 0, [])
    return rule_vocab


# if __name__ == "__main__":
#     rv, _ =create_rules()
#     print(rv)
#     print(rv.get_rule("Failed_Path").is_terminal)
