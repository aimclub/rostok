from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, Optional, Union, Type
from functools import singledispatchmethod
import rostok.block_builder_api.easy_body_shapes as easy_body_shapes
from rostok.block_builder_api.block_parameters import DefaultFrame, FrameTransform, JointInputType, Material


@dataclass
class BodyBlueprintType:
    """Use for mark block category
    """
    pass


@dataclass
class JointBlueprintType:
    """Use to mark block category
    """
    pass


@dataclass
class TransformBlueprintType:
    """Use to mark block category
    """
    pass


class TransformBlueprint(TransformBlueprintType):
    transform: FrameTransform = DefaultFrame
    is_transform_input = False


class RevolveJointBlueprint(JointBlueprintType):
    type_of_input: JointInputType = JointInputType.TORQUE
    radius: float = 0.07
    length: float = 0.4
    material: Material = Material()
    density = 10.0
    starting_angle = 0
    stiffness: float = 0.
    damping: float = 0.
    with_collision = True


class PrimitiveBodyBlueprint(BodyBlueprintType):
    shape: easy_body_shapes.ShapeTypes = easy_body_shapes.Box()
    density: float = 10.0
    material: Material = Material()
    is_collide: bool = True
    color: Optional[list[int]] = None


class EnvironmentBodyBlueprint(BodyBlueprintType):
    shape: easy_body_shapes.ShapeTypes = easy_body_shapes.Box()
    density: float = 10.0
    material: Material = Material()
    is_collide: bool = True
    color: Optional[list[int]] = None
    pos: FrameTransform = DefaultFrame


ALL_BLUEPRINT = Union[TransformBlueprint, RevolveJointBlueprint, PrimitiveBodyBlueprint,
                      EnvironmentBodyBlueprint]
ALL_BLUEPRINT_TYPE = Type[ALL_BLUEPRINT]

class NotImplementedErrorCreatorInterface(NotImplementedError):
    def __init__(self, *args: object) -> None:
        super().__init__('Need implementation for method in child class')

class BlockCreatorInterface():

    @classmethod
    @abstractmethod
    def create_transform(cls, blueprint: TransformBlueprint) -> Any:
        raise NotImplementedErrorCreatorInterface()

    @classmethod
    @abstractmethod
    def create_revolve_joint(cls, blueprint: RevolveJointBlueprint) -> Any:
        raise NotImplementedErrorCreatorInterface()

    @classmethod
    @abstractmethod
    def create_primitive_body(cls, blueprint: PrimitiveBodyBlueprint) -> Any:
        raise NotImplementedErrorCreatorInterface()
    @classmethod
    @abstractmethod
    def create_environment_body(cls, blueprint: EnvironmentBodyBlueprint) -> Any:
        raise NotImplementedErrorCreatorInterface()

    @singledispatchmethod
    @classmethod
    def init_block_from_blueprint(cls, blueprint) -> Any:
        raise (NotImplementedError(
            f'There is no implementation for class  {type(blueprint)}. In init_block_from_blueprint.'
        ))

    @init_block_from_blueprint.register
    @classmethod
    def _(cls, blueprint: TransformBlueprint):
        return cls.create_transform(blueprint)
    
    @init_block_from_blueprint.register
    @classmethod
    def _(cls, blueprint: RevolveJointBlueprint):
        return cls.create_revolve_joint(blueprint)
    
    @init_block_from_blueprint.register
    @classmethod
    def _(cls, blueprint: PrimitiveBodyBlueprint):
        return cls.create_primitive_body(blueprint)
    
    @init_block_from_blueprint.register
    @classmethod
    def _(cls, blueprint: EnvironmentBodyBlueprint):
        return cls.create_environment_body(blueprint)

