import networkx
import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
from enum import Enum
from abc import ABC
from abc import abstractmethod
from abc import abstractproperty
from collections import namedtuple
from typing import Optional
from collections import namedtuple


class BlockType(str, Enum):
    Transform = "Transform"
    Body = "Body"
    Bridge = "Bridge"


class Block(ABC):
    def __init__(self, builder):
        self.block_type = None

        self._ref_frame_in = None
        self._ref_frame_out = None
        self.transformed_frame_out = None

        self.builder: chrono.ChSystemNSC = builder
        self.is_build = False

    def apply_transform(self, in_block):
        pass


class BlockBridge(Block, ABC):
    def __init__(self, builder):
        super().__init__(builder)
        self.block_type = BlockType.Bridge


class BlockTransform(Block, ABC):
    def __init__(self, builder):
        super().__init__(builder)
        self.block_type = BlockType.Transform
        self.transform = None


class BlockBody(Block, ABC):
    def __init__(self, builder):
        super().__init__(builder)
        self.block_type = BlockType.Body
        self.body = None



"""

node (ref_frame_out)-> node_joint -> node(_ref_frame_in)

Короче тема такая 

Надо  сначала   зарендерить Body and Transform 
Попутно пременять трансформы к выходам parent блоков 
Даже жоинт хранит в себе выходной фрейм 
Потом отдельно жоинты
Отдельная функция для коннекта

"""


class ChronoBody(BlockBody):
    def __init__(self, builder, k):
        super().__init__(builder=builder)

        # Create body
        body = chrono.ChBody()
        self.body = body

        # Create shape
        # TODO setter for shape
        box_asset = chrono.ChBoxShape()
        box_asset.GetBoxGeometry().Size = chrono.ChVectorD(0.1, 0.5 * k, 0.1)

        self.body.AddAsset(box_asset)
        self.builder.Add(self.body)

        # Create markers aka RefFrames
        input_marker = chrono.ChMarker()
        out_marker = chrono.ChMarker()
        transformed_out_marker = chrono.ChMarker()

        input_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
        out_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
        transformed_out_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)

        self.body.AddMarker(input_marker)
        self.body.AddMarker(out_marker)
        self.body.AddMarker(transformed_out_marker)

        input_marker.SetPos(chrono.ChVectorD(0, -0.5 * k, 0))
        out_marker.SetPos(chrono.ChVectorD(0, 0.5 * k, 0))

        # Calc SetPos


        transformed_out_marker.SetCoord(out_marker.GetCoord())

        self._ref_frame_in = input_marker
        self._ref_frame_out = out_marker
        self.transformed_frame_out = transformed_out_marker

        #

    def move_to_out_frame(self, in_block: Block):
        self.builder.Update()
        local_coord_in_frame = self._ref_frame_in.GetCoord()
        abs_coord_out_frame = in_block.transformed_frame_out.GetAbsCoord()

        trans = chrono.ChFrameD(local_coord_in_frame)
        trans = trans.GetInverse()
        trans = trans.GetCoord()
        coord = abs_coord_out_frame * trans

        self.body.SetCoord(coord)


    def make_fix_joint(self, in_block):
        fix_joint = chrono.ChLinkMateFix()
        fix_joint.Initialize(in_block.body, self.body)
        self.builder.Add(fix_joint)

    def reset_transformed_frame_out(self):
        self.transformed_frame_out.SetCoord(self._ref_frame_out.GetCoord())

    def apply_transform(self, in_block: BlockTransform):
        self.reset_transformed_frame_out()
        frame_coord = self.transformed_frame_out.GetCoord()
        frame_coord = frame_coord * in_block.transform
        self.transformed_frame_out.SetCoord(frame_coord)

    @property
    def ref_frame_in(self):
        return self._ref_frame_in


class ChronoRevolveJoint(BlockBridge):
    class Axis(str, Enum):
        # Z is default rotation axis
        Z = chrono.ChQuaternionD(1, 0, 0, 0)
        Y = chrono.Q_ROTATE_Z_TO_Y
        X = chrono.Q_ROTATE_Z_TO_X

    def __init__(self, builder, axis=Axis.Z):
        super().__init__(builder=builder)
        self.joint = None
        self.axis = axis
        self._ref_frame_out = chrono.ChCoordsysD()

    def connect(self, in_block: ChronoBody, out_block: ChronoBody):
        # If we have two not initialize joints engine crash
        self.joint = chrono.ChLinkMotorRotationSpeed()
        self.builder.Add(self.joint)

        self.joint.Initialize(in_block.body, out_block.body, True,
                              in_block.transformed_frame_out, out_block.ref_frame_in)

    def apply_transform(self, in_block):
        self.transformed_frame_out = self._ref_frame_out * in_block.transform



class ChronoTransform(BlockTransform):
    def __init__(self, builder, transform: chrono.ChCoordsysD):
        super().__init__(builder=builder)
        self.transform = transform


def find_body_from_two_previous_blocks(sequence: list[Block], it: int) -> Optional[Block]:
    # b->t->j->t->b Longest sequence
    for i in reversed(range(it)[-2:]):
        if sequence[i].block_type == BlockType.Body:
            return sequence[i]
    return None


def find_body_from_two_after_blocks(sequence: list[Block], it: int) -> Optional[Block]:
    # b->t->j->t->b Longest sequence
    for block in sequence[it:it+2]:
        if block.block_type == BlockType.Body:
            return block
    return None


def build_branch(sequence: list[Block]):
    # Make body and apply transform
    previous_body_block = None
    need_fix_joint = False

    for it, block in enumerate(sequence):
        if block.block_type is BlockType.Body:
            # First body
            if previous_body_block is None:
                need_fix_joint = True
                previous_body_block = block
            else:
                block.move_to_out_frame(previous_body_block)
                if need_fix_joint:
                    block.make_fix_joint(previous_body_block)

                need_fix_joint = True
                previous_body_block = block

        elif block.block_type is BlockType.Bridge:
            need_fix_joint = False

        elif block.block_type is BlockType.Transform:
            sequence[it - 1].apply_transform(block)

        for it, block in enumerate(sequence):
            if block.block_type == BlockType.Bridge:
                block_in = find_body_from_two_previous_blocks(sequence, it)
                block_out = find_body_from_two_after_blocks(sequence, it)

                if(block_in is None):
                    raise Exception('Bridge element require', 'eggs')
                if(block_out is None):
                    raise Exception('spam', 'eggs')

                block.connect(block_in, block_out)




mysystem = chrono.ChSystemNSC()

bodik1 = ChronoBody(mysystem, k=1)
bodik2 = ChronoBody(mysystem, k=0.5)
bodik3 = ChronoBody(mysystem, k=0.5)
bodik4 = ChronoBody(mysystem, k=0.5)
bodik5 = ChronoBody(mysystem, k=1)

cordan = chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.ChQuaternionD(1, 0, 0, 0))
transform1 = ChronoTransform(mysystem, cordan)

cordan2 = chrono.ChCoordsysD(chrono.ChVectorD(0, 0.5, 0), chrono.Q_ROTATE_Z_TO_Y)
transform2 = ChronoTransform(mysystem, cordan2)

cordan3 = chrono.ChCoordsysD(chrono.ChVectorD(0, 0.0, 0), chrono.Q_ROTATE_Z_TO_Y)
transform3 = ChronoTransform(mysystem, cordan3)

col1 = chrono.ChColorAsset()
col1.SetColor(chrono.ChColor(0.6, 0, 0))
bodik1.body.AddAsset(col1)

col1 = chrono.ChColorAsset()
col1.SetColor(chrono.ChColor(0, 0.6, 0))
bodik2.body.AddAsset(col1)

col1 = chrono.ChColorAsset()
col1.SetColor(chrono.ChColor(0, 0.6, 0.9))
bodik3.body.AddAsset(col1)

col1 = chrono.ChColorAsset()
col1.SetColor(chrono.ChColor(0.9, 0.6, 0.9))
bodik4.body.AddAsset(col1)

col1 = chrono.ChColorAsset()
col1.SetColor(chrono.ChColor(0.9, 0.0, 0.9))
bodik5.body.AddAsset(col1)

bodik1.body.SetBodyFixed(True)

joint1 = ChronoRevolveJoint(mysystem)
sequa1 = [bodik1, transform1,joint1 ,bodik2, transform2, bodik3]
build_branch(sequa1)

joint2 = ChronoRevolveJoint(mysystem)
joint3 = ChronoRevolveJoint(mysystem)


sequa2 = [bodik2, transform3,joint2,bodik4,transform3,joint3,bodik5]
build_branch(sequa2)

myapplication = chronoirr.ChIrrApp(mysystem, 'PyChrono example', chronoirr.dimension2du(1024, 768))
myapplication.AddTypicalCamera(chronoirr.vector3df(0.6, 0.6, 0.6))
myapplication.AddTypicalLights()
myapplication.AssetBindAll()
myapplication.AssetUpdateAll()
myapplication.SetPlotLinkFrames(True)
myapplication.SetTimestep(0.005)
myapplication.SetTryRealtime(True)

while myapplication.GetDevice().run():
    mysystem.Update()
    myapplication.BeginScene(True, True, chronoirr.SColor(255, 140, 161, 192))
    myapplication.DrawAll()
    myapplication.DoStep()
    myapplication.EndScene()
