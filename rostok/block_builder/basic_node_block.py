from enum import Enum
from abc import ABC

import rostok.block_builder.body_size as bs

class SimpleBody(Enum):
    BOX: bs.BoxSize = bs.BoxSize(0.1,0.2,0.3)
    CYLINDER: bs.CylinderSize = bs.CylinderSize(0.1,2)
    SPHERE: bs.SphereSize = bs.SphereSize(0.1)
    ELLIPSOID: bs.EllipsoidSize = bs.EllipsoidSize(0.2,0.3,0.5)

class BlockType(str, Enum):
    TRANSFORM = "Transform"
    BODY = "Body"
    BRIDGE = "Bridge"


class Block(ABC):

    def __init__(self, builder):
        self.block_type = None

        self._ref_frame_in = None
        self._ref_frame_out = None
        self.transformed_frame_out = None

        self.builder = builder
        self.is_build = False

    def apply_transform(self, in_block):
        pass


class BlockBridge(Block, ABC):

    def __init__(self, builder):
        super().__init__(builder)
        self.block_type = BlockType.BRIDGE


class BlockTransform(Block, ABC):

    def __init__(self, builder):
        super().__init__(builder)
        self.block_type = BlockType.TRANSFORM
        self.transform = None


class BlockBody(Block, ABC):

    def __init__(self, builder):
        super().__init__(builder)
        self.block_type = BlockType.BODY
        self.body = None


class RobotBody(ABC):

    def __init__(self):
        pass
