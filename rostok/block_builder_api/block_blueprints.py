from abc import abstractmethod
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Generic, Optional, Type, Union

from traitlets import Bool

import rostok.block_builder_api.easy_body_shapes as easy_body_shapes
from rostok.utils.dataset_materials.material_dataclass_manipulating import DefaultChronoMaterialNSC, Material
from rostok.block_builder_api.block_parameters import (DefaultFrame,
                                                       FrameTransform,
                                                       JointInputType,
                                                       )


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


@dataclass
class TransformBlueprint(TransformBlueprintType):
    transform: FrameTransform = DefaultFrame
    is_transform_input = False


@dataclass
class RevolveJointBlueprint(JointBlueprintType):
    type_of_input: JointInputType = JointInputType.TORQUE
    starting_angle: float = 0.
    stiffness: float = 0.
    damping: float = 0.
    equilibrium_position:float = 0.
    stopper: Optional[list[float]] = None


class RevolveJointBlueprintWithBody(JointBlueprintType):
    type_of_input: JointInputType = JointInputType.TORQUE
    radius: float = 0.015
    length: float = 0.03
    material: Material = DefaultChronoMaterialNSC()
    density: float= 400
    starting_angle: float = 0.
    stiffness: float = 0.
    damping: float = 0.
    offset: float = 0.
    equilibrium_position:float = 0.
    with_collision: bool = True
    stopper: Optional[list[float]] = None


@dataclass
class PrimitiveBodyBlueprint(BodyBlueprintType):
    shape: easy_body_shapes.ShapeTypes = easy_body_shapes.Box()
    density: float = 400
    material: Material = DefaultChronoMaterialNSC()
    is_collide: bool = True
    color: Optional[list[int]] = None


@dataclass
class EnvironmentBodyBlueprint(BodyBlueprintType):
    shape: easy_body_shapes.ShapeTypes = easy_body_shapes.Box()
    density: float = 100
    material: Material = DefaultChronoMaterialNSC()
    is_collide: bool = True
    color: Optional[list[int]] = None
    pos: FrameTransform = DefaultFrame
    
    def __hash__(self) -> int:
        atrb = [self.shape, self.density, self.material, self.is_collide, self.pos]
        return hash(("EnvironmentBodyBlueprint".join([str(i) for i in atrb])))


ALL_BLUEPRINT = Union[TransformBlueprint, RevolveJointBlueprint, PrimitiveBodyBlueprint,
                      EnvironmentBodyBlueprint]
ALL_BLUEPRINT_TYPE = Type[ALL_BLUEPRINT]


class NotImplementedErrorCreatorInterface(NotImplementedError):

    def __init__(self, *args: object) -> None:
        super().__init__('Need implementation for method in child class')


class BlockCreatorInterface:
    """ To use it, you need to implement functions for creating from blueprints.

    Raises:
        NotImplementedErrorCreatorInterface: Need implementation for method in child class

    """

    @classmethod
    @abstractmethod
    def create_transform(cls, blueprint: TransformBlueprint):
        raise NotImplementedErrorCreatorInterface()

    @classmethod
    @abstractmethod
    def create_revolve_joint(cls, blueprint: RevolveJointBlueprint):
        raise NotImplementedErrorCreatorInterface()

    @classmethod
    @abstractmethod
    def create_revolve_joint_with_body(cls, blueprint: RevolveJointBlueprintWithBody):
        raise NotImplementedErrorCreatorInterface()

    @classmethod
    @abstractmethod
    def create_primitive_body(cls, blueprint: PrimitiveBodyBlueprint):
        raise NotImplementedErrorCreatorInterface()

    @classmethod
    @abstractmethod
    def create_environment_body(cls, blueprint: EnvironmentBodyBlueprint):
        raise NotImplementedErrorCreatorInterface()

    @singledispatchmethod
    @classmethod
    def init_block_from_blueprint(cls, blueprint):
        """ Make mapping creation functions with blueprints

        Args:
            blueprint : Any blueprint

        """
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
    def _(cls, blueprint: RevolveJointBlueprintWithBody):
        return cls.create_revolve_joint(blueprint)

    @init_block_from_blueprint.register
    @classmethod
    def _(cls, blueprint: PrimitiveBodyBlueprint):
        return cls.create_primitive_body(blueprint)

    @init_block_from_blueprint.register
    @classmethod
    def _(cls, blueprint: EnvironmentBodyBlueprint):
        return cls.create_environment_body(blueprint)
