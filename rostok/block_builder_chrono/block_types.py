from abc import ABC
from enum import Enum


class BlockType(str, Enum):
    TRANSFORM = "Transform"
    SELFTRANSFORM = "SelfTransform"
    BODY = "Body"
    BRIDGE = "Bridge"


class Block(ABC):
    def __init__(self):
        self.block_type = None
        self.is_build = False


class BlockBridge(Block, ABC):
    def __init__(self):
        super().__init__()
        self.block_type = BlockType.BRIDGE


class BlockTransform(Block, ABC):
    def __init__(self, is_selftransform:bool = False):
        super().__init__()
        if is_selftransform:
            self.block_type = BlockType.SELFTRANSFORM
        else:
            self.block_type = BlockType.TRANSFORM


class BlockBody(Block, ABC):
    def __init__(self):
        super().__init__()
        self.block_type = BlockType.BODY
        self.body = None
        self._ref_frame_in = None
        self._ref_frame_out = None
        self.transformed_frame_out = None
        self.transformed_frame_input = None

    def apply_transform_out(self, transform):
        pass

    def apply_transform_input(self, transform):
        pass
