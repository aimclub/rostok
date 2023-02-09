from enum import Enum
from abc import ABC


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
