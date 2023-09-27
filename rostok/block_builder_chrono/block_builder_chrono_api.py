from copy import deepcopy

from rostok.block_builder_api.block_blueprints import (
    BlockCreatorInterface, EnvironmentBodyBlueprint, PrimitiveBodyBlueprint,
    RevolveJointBlueprint, TransformBlueprint)
from rostok.block_builder_chrono.adapt_block_blueprint import \
    convert_joint_input_type_to_chrono
from rostok.block_builder_chrono.block_classes import (ChronoEasyShapeObject,
                                                       ChronoRevolveJoint,
                                                       ChronoTransform,
                                                       PrimitiveBody)


class ChronoBlockCreatorInterface(BlockCreatorInterface):

    @classmethod
    def create_transform(cls, blueprint: TransformBlueprint) -> ChronoTransform:
        return ChronoTransform(blueprint.transform, blueprint.is_transform_input)

    @classmethod
    def create_revolve_joint(cls, blueprint: RevolveJointBlueprint) -> ChronoRevolveJoint:
        blueprint_chrono = deepcopy(blueprint)
        type_of_input_blue = blueprint_chrono.type_of_input
        material_blue = blueprint_chrono.material
        type_of_input_chrono = convert_joint_input_type_to_chrono(type_of_input_blue)
        # This works because the name of initialization arguments is equal to Blueprint
        blueprint_chrono.type_of_input = type_of_input_chrono  # type: ignore
        blueprint_chrono.material = material_blue  # type: ignore
        return ChronoRevolveJoint(**blueprint_chrono.__dict__)

    @classmethod
    def create_primitive_body(cls, blueprint: PrimitiveBodyBlueprint) -> PrimitiveBody:
        blueprint_chrono = deepcopy(blueprint)
        material_blue = blueprint_chrono.material
        # This works because the name of initialization arguments is equal to Blueprint
        blueprint_chrono.material = material_blue  # type: ignore
        return PrimitiveBody(**blueprint_chrono.__dict__)

    @classmethod
    def create_environment_body(cls, blueprint: EnvironmentBodyBlueprint) -> ChronoEasyShapeObject:
        blueprint_chrono = deepcopy(blueprint)
        material_blue = blueprint_chrono.material
        # This works because the name of initialization arguments is equal to Blueprint
        blueprint_chrono.material = material_blue  # type: ignore
        return ChronoEasyShapeObject(**blueprint_chrono.__dict__)