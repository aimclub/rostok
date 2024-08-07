import numpy as np
import pychrono as chrono

from rostok.graph_grammar.node import GraphGrammar
from rostok.block_builder_api.block_blueprints import (PrimitiveBodyBlueprint,
                                                       RevolveJointBlueprint,
                                                       TransformBlueprint)
from rostok.block_builder_api.block_parameters import JointInputType
from rostok.block_builder_api.easy_body_shapes import Box, Cylinder
from rostok.block_builder_chrono.blocks_utils import FrameTransform
from rostok.graph_grammar import rule_vocabulary
from rostok.graph_grammar.node import ROOT
from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.utils.dataset_materials.material_dataclass_manipulating import (
    DefaultChronoMaterialNSC, DefaultChronoMaterialSMC)


def get_density_box(mass: float, box: Box):
    volume = box.height_z * box.length_y * box.width_x
    return mass / volume

def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]

def rotation_y(alpha):
    quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
    return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]

def rotation_z(alpha):
    quat_Z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
    return [quat_Z_ang_alpha.e0, quat_Z_ang_alpha.e1, quat_Z_ang_alpha.e2, quat_Z_ang_alpha.e3]

def create_rules(smc=False):

    if smc:
        def_mat = DefaultChronoMaterialSMC()
    else:
        def_mat = DefaultChronoMaterialNSC()
    # blueprint for the palm
    main_body = PrimitiveBodyBlueprint(
        Box(0.5, 0.1, 0.3), material=def_mat, color=[255, 0, 0], density=1000)

    wheel_body = PrimitiveBodyBlueprint(
        Cylinder(0.03, 0.01), material=def_mat, color=[0, 120, 255], density=1000)

    # blueprint for the base
    base = PrimitiveBodyBlueprint(Box(0.02, 0.03, 0.02),
                                  material=def_mat,
                                  color=[120, 120, 0],
                                  density=1000)
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


    x_translation_values = [0.07, 0.107, 0.144]
    X_TRANSLATIONS_POSITIVE = list(
        map(lambda x: FrameTransform([x, 0, 0], [1, 0, 0, 0]), x_translation_values))
    X_TRANSLATIONS_NEGATIVE=list(
        map(lambda x: FrameTransform([-x, 0, 0], [1, 0, 0, 0]), x_translation_values))
    z_translation_values = [0.055, 0.152, 0.19]
    Z_TRANSLATIONS_POSITIVE = list(
        map(lambda x: FrameTransform([0, 0, x], [1, 0, 0, 0]), z_translation_values))
    Z_TRANSLATIONS_NEGATIVE = list(
        map(lambda x: FrameTransform([0, 0, -x], [1, 0, 0, 0]), z_translation_values))


    turn_values = [90]
    TURNS_POSITIVE = list(map(lambda x: FrameTransform(
        [0, 0, 0], rotation_x(x)), turn_values))
    TURNS_NEGATIVE = list(map(lambda x: FrameTransform(
        [0, 0, 0], rotation_x(-x)), turn_values))

    # create transform blueprints from the values
    x_translation_positive_transform = list(
        map(lambda x: TransformBlueprint(x), X_TRANSLATIONS_POSITIVE))
    x_translation_negative_transform = list(
        map(lambda x: TransformBlueprint(x), X_TRANSLATIONS_NEGATIVE))
    z_translation_positive_transforms = list(
        map(lambda x: TransformBlueprint(x), Z_TRANSLATIONS_POSITIVE))
    z_translation_negative_transforms = list(
        map(lambda x: TransformBlueprint(x), Z_TRANSLATIONS_NEGATIVE))

    positive_turn_transform = list(
        map(lambda x: TransformBlueprint(x), TURNS_POSITIVE))
    negative_turn_transform = list(
        map(lambda x: TransformBlueprint(x), TURNS_NEGATIVE))


    stiffness__values_base = [0.9, 0.095, 0.07]
    preload_angle_values_base = [0, 0, 0]
    no_control_base = list(
        map(
            lambda x, y: RevolveJointBlueprint(JointInputType.UNCONTROL,
                                               stiffness=x,
                                               damping=0.02,
                                               equilibrium_position=y), stiffness__values_base,
            preload_angle_values_base))
    revolve = RevolveJointBlueprint(JointInputType.TORQUE,  stiffness=0, damping=0.0)
    # Nodes
    node_vocab = NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node(label="MB", is_terminal=True,
                           block_blueprint=main_body)
    node_vocab.create_node(label="W", is_terminal=True,block_blueprint=wheel_body)
    node_vocab.create_node(label="M", is_terminal=True, block_blueprint=revolve)
    node_vocab.create_node(label="G")
    node_vocab.create_node(label="L")
    node_vocab.create_node(label="XT")
    node_vocab.create_node(label="ZT")
    node_vocab.create_node(label="NXT")
    node_vocab.create_node(label="NZT")

    node_vocab.create_node(label="B", is_terminal=True, block_blueprint=base)
    node_vocab.create_node(label="TP", is_terminal=True,block_blueprint=positive_turn_transform[0])
    node_vocab.create_node(label="TN", is_terminal=True,block_blueprint=negative_turn_transform[0])

    node_vocab.create_node(label="XT0", is_terminal=True,block_blueprint=x_translation_positive_transform[0])
    node_vocab.create_node(label="XT1", is_terminal=True,block_blueprint=x_translation_positive_transform[1])
    node_vocab.create_node(label="XT2", is_terminal=True,block_blueprint=x_translation_positive_transform[2])

    node_vocab.create_node(label="ZT0", is_terminal=True,block_blueprint=z_translation_positive_transforms[0])
    node_vocab.create_node(label="ZT1", is_terminal=True,block_blueprint=z_translation_positive_transforms[1])
    node_vocab.create_node(label="ZT2", is_terminal=True,block_blueprint=z_translation_positive_transforms[2])

    node_vocab.create_node(label="NXT0", is_terminal=True,block_blueprint=x_translation_negative_transform[0])
    node_vocab.create_node(label="NXT1", is_terminal=True,block_blueprint=x_translation_negative_transform[1])
    node_vocab.create_node(label="NXT2", is_terminal=True,block_blueprint=x_translation_negative_transform[2])

    node_vocab.create_node(label="NZT0", is_terminal=True,block_blueprint=z_translation_negative_transforms[0])
    node_vocab.create_node(label="NZT1", is_terminal=True,block_blueprint=z_translation_negative_transforms[1])
    node_vocab.create_node(label="NZT2", is_terminal=True,block_blueprint=z_translation_negative_transforms[2])
    node_vocab.create_node(label="J", is_terminal=True, block_blueprint=no_control_base[0])

    for i, bp in enumerate(link):
        node_vocab.create_node(label=f"L{i}", is_terminal=True, block_blueprint=bp)

    print(node_vocab.terminal_node_dict)
    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)
    rule_vocab.create_rule("Init", ["ROOT"], ["MB"], 0, 0,[])
    rule_vocab.create_rule("Add_FR_W", ["MB"], ["MB","XT","ZT","B", "G"], 0, 0,[(0, 1), (1, 2),
                                                                                    (2, 3), (3, 4)])
    rule_vocab.create_rule("Add_FL_W", ["MB"], ["MB","XT","NZT","B", "G"], 0, 0,[(0, 1), (1, 2),
                                                                                    (2, 3), (3, 4)])
    
    rule_vocab.create_rule("Add_BR_W", ["MB"], ["MB","NXT","ZT","B", "G"], 0, 0,[(0, 1), (1, 2),
                                                                                    (2, 3), (3, 4)])
    rule_vocab.create_rule("Add_BL_W", ["MB"], ["MB","NXT","NZT","B", "G"], 0, 0,[(0, 1), (1, 2),
                                                                                    (2, 3), (3, 4)])

    rule_vocab.create_rule("Extension",["G"],["J","L", "G"], 0,2,[(0, 1), (1, 2)])

    rule_vocab.create_rule('Stop_Extension',["G"], [],0,0,[])

    rule_vocab.create_rule("Wheel",["G"], ["M","W"],0,0,[(0,1)])
    rule_vocab.create_rule("Terminal_Positive_XTranslate_0", [
                           "XT"], ["XT0"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_XTranslate_1", [
                           "XT"], ["XT1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_XTranslate_2", [
                           "XT"], ["XT2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_ZTranslate_0", [
                           "ZT"], ["ZT0"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_ZTranslate_1", [
                           "ZT"], ["ZT1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Positive_ZTranslate_2", [
                           "ZT"], ["ZT2"], 0, 0, [])
    
    rule_vocab.create_rule("Terminal_Negative_XTranslate_0", [
                           "NXT"], ["NXT0"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_XTranslate_1", [
                           "NXT"], ["NXT1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_XTranslate_2", [
                           "NXT"], ["NXT2"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_ZTranslate_0", [
                           "NZT"], ["NZT0"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_ZTranslate_1", [
                           "NZT"], ["NZT1"], 0, 0, [])
    rule_vocab.create_rule("Terminal_Negative_ZTranslate_2", [
                           "NZT"], ["NZT2"], 0, 0, [])

    for i, bp in enumerate(link):
        rule_vocab.create_rule(f"Terminal_Link{i}", [
                           "L"], [f"L{i}"], 0, 0, [])
    return rule_vocab

def get_four_wheels():
    graph = GraphGrammar()
    rules = ["Init",
            'Add_FR_W',"Extension","Extension","Wheel",'Terminal_Positive_XTranslate_2', "Terminal_Positive_ZTranslate_1","Terminal_Link2","Terminal_Link1",
            "Add_FL_W", "Extension","Extension",'Terminal_Positive_XTranslate_2', "Terminal_Negative_ZTranslate_1","Wheel", "Terminal_Link2","Terminal_Link1",
            "Add_BR_W", "Extension","Extension",'Terminal_Negative_XTranslate_2', "Terminal_Positive_ZTranslate_1","Wheel", "Terminal_Link2","Terminal_Link1",
            "Add_BL_W", "Extension","Extension",'Terminal_Negative_XTranslate_2', "Terminal_Negative_ZTranslate_1","Wheel", "Terminal_Link2","Terminal_Link1",
    ]
    rule_vocabul = create_rules()
    for rule in rules:
        graph.apply_rule(rule_vocabul.get_rule(rule))

    return graph