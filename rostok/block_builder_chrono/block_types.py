from abc import ABC
from enum import Enum
from typing import Generic, TypeVar


Descriptor = TypeVar("Descriptor")


class BlockType(str, Enum):
    TRANSFORM_OUT = "Transform_Out"
    TRANSFORM_INPUT = "Transform_Input"
    BODY = "Body"
    BRIDGE = "Bridge"


class Block(Generic[Descriptor]):

    def __init__(self):
        self.block_type = None
        self.is_build = False

    @classmethod
    def initialize_from_descriptor(cls, des: Descriptor):
        return cls(**des.__dict__)


class BlockBridge(Block[Descriptor]):

    def __init__(self):
        super().__init__()
        self.block_type = BlockType.BRIDGE


class BlockTransform(Block[Descriptor]):

    def __init__(self, is_transform_input: bool = False):
        super().__init__()
        if is_transform_input:
            self.block_type = BlockType.TRANSFORM_INPUT
        else:
            self.block_type = BlockType.TRANSFORM_OUT


class BlockBody(Block[Descriptor]):

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
