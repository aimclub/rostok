import numpy as np
import pychrono as chrono

from rostok.block_builder_api.block_blueprints import (PrimitiveBodyBlueprint,
                                                       RevolveJointBlueprint,
                                                       TransformBlueprint)
from rostok.block_builder_api.block_parameters import JointInputType
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_chrono.blocks_utils import FrameTransform
from rostok.graph_grammar import rule_vocabulary
from rostok.graph_grammar.node import ROOT
from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.utils.dataset_materials.material_dataclass_manipulating import (
    DefaultChronoMaterialNSC, DefaultChronoMaterialSMC)


def get_density_box(mass: float, box: Box):
    volume = box.height_z * box.length_y * box.width_x
    return mass / volume


def rotation_y(alpha):
    quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
    return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]


def rotation_z(alpha):
    quat_Z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
    return [quat_Z_ang_alpha.e0, quat_Z_ang_alpha.e1, quat_Z_ang_alpha.e2, quat_Z_ang_alpha.e3]


import numpy as np
import pychrono as chrono

from rostok.block_builder_api.block_blueprints import (PrimitiveBodyBlueprint,
                                                       RevolveJointBlueprint,
                                                       TransformBlueprint)
from rostok.block_builder_api.block_parameters import JointInputType
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_chrono.blocks_utils import FrameTransform
from rostok.graph_grammar import rule_vocabulary
from rostok.graph_grammar.node import ROOT
from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.utils.dataset_materials.material_dataclass_manipulating import (
    DefaultChronoMaterialNSC, DefaultChronoMaterialSMC)


def get_density_box(mass: float, box: Box):
    volume = box.height_z * box.length_y * box.width_x
    return mass / volume


def rotation_y(alpha):
    quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
    return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]


def rotation_z(alpha):
    quat_Z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
    return [quat_Z_ang_alpha.e0, quat_Z_ang_alpha.e1, quat_Z_ang_alpha.e2, quat_Z_ang_alpha.e3]


def create_rules(tendon=True, smc=False):

    if smc:
        def_mat = DefaultChronoMaterialSMC()
    else:
        def_mat = DefaultChronoMaterialNSC()
    # blueprint for the palm
    body = PrimitiveBodyBlueprint(
        Box(0.3, 0.3, 0.3), material=def_mat, color=[255, 0, 0])
    # blueprint for the base
    base = PrimitiveBodyBlueprint(Box(0.02, 0.05, 0.02),
                                  material=def_mat,
                                  color=[0, 120, 255],
                                  density=10000)
    

    foot = PrimitiveBodyBlueprint(Box(0.1, 0.05, 0.02),
                                  material=def_mat,
                                  color=[0, 120, 255],
                                  density=100)
    # sets effective density for the links, the real links are considered to be extendable.
    link_mass = (28 + 1.62 + 2.77) * 1e-3
    length_link = [0.05, 0.06, 0.075]
    # create link blueprints using mass and length parameters
    link = list(
        map(
            lambda x: PrimitiveBodyBlueprint(Box(0.02, x, 0.02),
                                             material=def_mat,
                                             color=[0, 120, 255],
                                             density=get_density_box(link_mass, Box(
                                                 0.02, x, 0.02))), length_link))
    dummy_link = PrimitiveBodyBlueprint(Box(0.01,0.000001,0.01), material=def_mat, density =0.0001)


    x_translation_values = [0.07, 0.107, 0.144]
    X_TRANSLATIONS = list(
        map(lambda x: FrameTransform([x, 0, 0], [1, 0, 0, 0]), x_translation_values))

    z_translation_values = [0.055, 0.092, 0.129]
    Z_TRANSLATIONS_POSITIVE = list(
        map(lambda x: FrameTransform([0, 0, x], [1, 0, 0, 0]), z_translation_values))
    Z_TRANSLATIONS_NEGATIVE = list(
        map(lambda x: FrameTransform([0, 0, -x], [1, 0, 0, 0]), z_translation_values))

    # rotation to 180 degrees around vertical axis
    REVERSE_Y = FrameTransform([0, 0, 0], [0, 0, 1, 0])

    turn_values = [0, 30, 60]
    TURNS_POSITIVE = list(map(lambda x: FrameTransform(
        [0, 0, 0], rotation_y(x)), turn_values))
    TURNS_NEGATIVE = list(map(lambda x: FrameTransform(
        [0, 0, 0], rotation_y(-x)), turn_values))
    JOINT_TURN_POSITIVE = FrameTransform([0,0,0], rotation_y(90))
    JOINT_TURN_NEGATIVE = FrameTransform([0,0,0], rotation_y(-90))
    joint_turn_positive = TransformBlueprint(JOINT_TURN_POSITIVE)
    joint_turn_negative = TransformBlueprint(JOINT_TURN_NEGATIVE)
    # create transform blueprints from the values
    reverse_transform = TransformBlueprint(REVERSE_Y)
    mass_joint = (10 / 3 + 0.51 * 2 + 0.64 + 1.3) * 1e-3  # 0.012
    joint_radius_base = 0.015
    joint_radius = 0.015
    joint_length = 0.025
    density_joint = (mass_joint / (0.03 * 3.14 * joint_radius**2))

    # stiffness is a coefficient that couples torque to angle of rotation
    # preload is represented as starting angle of the spring
    if tendon:
        stiffness_values = [0.095, 0.07]
        preload_angle_values = [0.8, 0.8]
    else:
        stiffness_values = [0, 0.]
        preload_angle_values = [0., 0.]
    stiffness_values = [0, 0.]
    preload_angle_values = [0., 0.]
    no_control = list(
        map(
            lambda x, y: RevolveJointBlueprint(JointInputType.UNCONTROL,
                                               stiffness=x,
                                               damping=0.01,
                                               offset=0.008,
                                               material=def_mat,
                                               radius=joint_radius,
                                               length=joint_length,
                                               density=density_joint,
                                               equilibrium_position=y), stiffness_values,
            preload_angle_values))

    stiffness__values_base = [0.19, 0.095, 0.07]
    preload_angle_values_base = [0, 0, 0]
    no_control_base = list(
        map(
            lambda x, y: RevolveJointBlueprint(JointInputType.UNCONTROL,
                                               stiffness=x,
                                               damping=0.01,
                                               offset=0,
                                               material=def_mat,
                                               radius=joint_radius_base,
                                               length=joint_length,
                                               density=density_joint,
                                               equilibrium_position=y), stiffness__values_base,
            preload_angle_values_base))
    revolve = RevolveJointBlueprint(JointInputType.TORQUE, material=def_mat, radius=joint_radius_base,
                                    length=joint_length, density=density_joint, stiffness=0.0, damping=0)
    revolve_base = RevolveJointBlueprint(JointInputType.TORQUE, material=def_mat, radius=joint_radius_base,
                                         length=joint_length, density=density_joint, stiffness=0.0, damping=0)
    # Nodes
    node_vocab = NodeVocabulary()
    node_vocab.add_node(ROOT)

    node_vocab.create_node(label="BO", is_terminal=True,
                           block_blueprint=body)
    
    node_vocab.create_node(label="LBL") # Left_Biped_Leg

    node_vocab.create_node(label="RBL") # Right_Biped_Leg
    
    node_vocab.create_node(label="LFQL") # Left_Front_Quadruped_Leg
    node_vocab.create_node(label="RFQL") # Right_Front_Quadruped_Leg
    node_vocab.create_node(label="LHQL") # Left_Hind_Quadruped_Leg
    node_vocab.create_node(label="RHQL") # Right_Hind_Quadruped_Leg

    node_vocab.create_node(label="RE", is_terminal=True,
                           block_blueprint=reverse_transform) # Reverse
    node_vocab.create_node(label="JT", is_terminal=True,
                           block_blueprint=joint_turn_positive)
    node_vocab.create_node(label="RJT", is_terminal=True,
                           block_blueprint=joint_turn_negative)
    node_vocab.create_node(label="RT") 
    node_vocab.create_node(label="DB", is_terminal=True, block_blueprint=dummy_link)
    node_vocab.create_node(label="FT", is_terminal=True,
                           block_blueprint=foot)
    node_vocab.create_node(label="RT1",
                           is_terminal=True,
                           block_blueprint=x_translation_transform[0])
    node_vocab.create_node(label="RT2",
                           is_terminal=True,
                           block_blueprint=x_translation_transform[1])
    node_vocab.create_node(label="RT3",
                           is_terminal=True,
                           block_blueprint=x_translation_transform[2])
    node_vocab.create_node(label="FG")
    node_vocab.create_node(label="FG1")
    if tendon:
        node_vocab.create_node(label="J")
        node_vocab.create_node(label="J_1", is_terminal=True,
                               block_blueprint=no_control[0])
        node_vocab.create_node(label="J_2", is_terminal=True,
                               block_blueprint=no_control[1])

        node_vocab.create_node(label="JB")
        node_vocab.create_node(
            label="JB_1", is_terminal=True, block_blueprint=no_control_base[0])
        node_vocab.create_node(
            label="JB_2", is_terminal=True, block_blueprint=no_control_base[1])
        node_vocab.create_node(
            label="JB_3", is_terminal=True, block_blueprint=no_control_base[2])
    else:
        node_vocab.create_node(
            label="J", is_terminal=True, block_blueprint=revolve)
        node_vocab.create_node(label="JB", is_terminal=True,
                               block_blueprint=revolve_base)
    node_vocab.create_node(label="L")
    node_vocab.create_node(label="B", is_terminal=True, block_blueprint=base)
    node_vocab.create_node(label="L1", is_terminal=True,
                           block_blueprint=link[0])
    node_vocab.create_node(label="L2", is_terminal=True,
                           block_blueprint=link[1])
    node_vocab.create_node(label="L3", is_terminal=True,
                           block_blueprint=link[2])

    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

    rule_vocab.create_rule("Init_Biped", ["ROOT"], ["BO", "LBL", "RBL"], 0, 0,
                           [(0, 1), (0, 2)])

    rule_vocab.create_rule("AddFinger", ["F"], ["RT", "B", "JB", "L", "FG"], 0, 0, [(0, 1), (1, 2),
                                                                                    (2, 3), (3, 4)])
    rule_vocab.create_rule("AddLeftLeg", ["LBL"], ["TN", "B", "JT", "J","DB","RJT","FG"],0,0,[(0, 1), (1, 2),
                                                                                    (2, 3), (3, 4),(4,5),(5,6)])
    rule_vocab.create_rule("AddRightLeg", ["RBL"], ["TP", "B", "JT", "J","DB","RJT","FG"],0,0,[(0, 1), (1, 2),
                                                                                    (2, 3), (3, 4),(4,5),(5,6)])
    


    rule_vocab.create_rule("Terminal_Radial_Translate1",
                           ["RT"], ["RT1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Radial_Translate2",
                           ["RT"], ["RT2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Radial_Translate3",
                           ["RT"], ["RT3"], 0, 0, [])

    rule_vocab.create_rule("Phalanx", ["FG"], [
                           "J", "L", "FG"], 0, 0, [(0, 1), (1, 2)])
    # rule_vocab.create_rule("Phalanx_1", ["FG1"], ["B", "JB", "L", "FG"], 0, 0, [(0, 1), (1, 2), (2, 3)])
    #
    if tendon:
        rule_vocab.create_rule('Terminal_Joint_1', ['J'], ["J_1"], 0, 0, [])
        rule_vocab.create_rule('Terminal_Joint_2', ['J'], ["J_2"], 0, 0, [])

        rule_vocab.create_rule('Terminal_Base_Joint_1', [
                               'JB'], ["JB_1"], 0, 0, [])
        rule_vocab.create_rule('Terminal_Base_Joint_2', [
                               'JB'], ["JB_2"], 0, 0, [])
        rule_vocab.create_rule('Terminal_Base_Joint_3', [
                               'JB'], ["JB_3"], 0, 0, [])

    rule_vocab.create_rule("Terminal_Link1", ["L"], ["L1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Link2", ["L"], ["L2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Link3", ["L"], ["L3"], 0, 0, [])
    #rule_vocab.create_rule("Remove_FG", ["FG"], [], 0, 0, [])
    rule_vocab.create_rule("AddFoot", ["FG"], ["J", "FT"], 0, 0, [(0, 1), (1, 2)])
    # rule_vocab.create_rule("Remove_FG1", ["FG1"], [], 0, 0, [])

    rule_vocab.create_rule("Terminal_Positive_Translate1", [
                           "TP"], ["TP1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Translate2", [
                           "TP"], ["TP2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Translate3", [
                           "TP"], ["TP3"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Translate1", [
                           "TN"], ["TN1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Translate2", [
                           "TN"], ["TN2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Translate3", [
                           "TN"], ["TN3"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Turn_0", [
                           "TURN_P"], ["TURN_P_0"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Turn_1", [
                           "TURN_P"], ["TURN_P_1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_Turn_2", [
                           "TURN_P"], ["TURN_P_2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Turn_0", [
                           "TURN_N"], ["TURN_N_0"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Turn_1", [
                           "TURN_N"], ["TURN_N_1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_Turn_2", [
                           "TURN_N"], ["TURN_N_2"], 0, 0, [])
    return rule_vocab

from rostok.graph_grammar.node import GraphGrammar

def get_biped():
    graph = GraphGrammar()
    rules = [
        "Init_Biped", "AddLeftLeg", "AddRightLeg", "Phalanx", "Phalanx", "AddFoot", "AddFoot", "Terminal_Link2", "Terminal_Link2"
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph
# if __name__ == "__main__":
#     rv, _ =create_rules()
#     print(rv)
#     print(rv.get_rule("Failed_Path").is_terminal)
