import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
from enum import Enum
from abc import ABC
from typing import Optional
import random


class BlockType(str, Enum):
    Transform = "Transform"
    Body = "Body"
    Bridge = "Bridge"

class SpringTorque(chrono.TorqueFunctor):
    def __init__(self, spring_coef, damping_coef, rest_angle):
        super(SpringTorque, self).__init__()
        self.spring_coef = spring_coef
        self.damping_coef = damping_coef
        self.rest_angle = rest_angle

    def evaluate(self,  #
                 time,  # current time
                 angle,  # relative angle of rotation
                 vel,  # relative angular speed
                 link):  # back-pointer to associated link
        torque = 0
        if self.spring_coef > 10**-3:
            torque = -self.spring_coef * (angle - self.rest_angle) - self.damping_coef * vel
        else:
            torque = - self.damping_coef * vel
        return torque

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


class ChronoBody(BlockBody):
    def __init__(self, builder, length=1, width=0.1, random_color=True, mass=1, material = None):
        super().__init__(builder=builder)

        # Create body
        self.body = chrono.ChBody()
        self.body.SetMass(mass)

        # Create shape
        # TODO setter for shape
        box_asset = chrono.ChBoxShape()
        box_asset.GetBoxGeometry().Size = chrono.ChVectorD(width, length, width)
        
        if material:
            self.body.GetCollisionModel().ClearModel()
            self.body.GetCollisionModel().AddBox(material,width,length,width)
            self.body.GetCollisionModel().BuildModel()
            

        self.body.AddVisualShape(box_asset)
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

        input_marker.SetPos(chrono.ChVectorD(0, -length, 0))
        out_marker.SetPos(chrono.ChVectorD(0, length, 0))

        # Calc SetPos
        transformed_out_marker.SetCoord(out_marker.GetCoord())

        self._ref_frame_in = input_marker
        self._ref_frame_out = out_marker
        self.transformed_frame_out = transformed_out_marker

        if random_color:
            rgb = [random.random(), random.random(), random.random()]
            rgb[int(random.random() * 2)] *= 0.2
            self.body.GetVisualShape(0).SetColor(chrono.ChColor(*rgb))

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
    # Variants of joint control
    class InputType(str,Enum):
        Torque = {"Name":"Torque","TypeMotor":chrono.ChLinkMotorRotationTorque} 
        Velocity = {"Name":"Speed" ,"TypeMotor":chrono.ChLinkMotorRotationSpeed}
        Position = {"Name":"Angle" ,"TypeMotor":chrono.ChLinkMotorRotationAngle}
        Uncontrol = {"Name":"Uncontrol" ,"TypeMotor":chrono.ChLinkRevolute}

        def __init__(self,vals):
            self.num = vals["Name"]
            self.motor = vals["TypeMotor"]

    class Axis(str, Enum):
        # Z is default rotation axis
        Z = chrono.ChQuaternionD(1, 0, 0, 0)
        Y = chrono.Q_ROTATE_Z_TO_Y
        X = chrono.Q_ROTATE_Z_TO_X

    def __init__(self, builder, axis=Axis.Z, type_of_input = InputType.Position,
                 stiffness=0, damping=0, equilibrium_position=0):
        super().__init__(builder=builder)
        self.joint = None
        self.axis = axis
        self.input_type = type_of_input
        self._ref_frame_out = chrono.ChCoordsysD()
        # Spring Damper params
        self.joint_spring = None
        self.torque_functor = None
        self.stiffness = stiffness
        self.damping = damping
        self.equilibrium_position = equilibrium_position

    def connect(self, in_block: ChronoBody, out_block: ChronoBody):
        # If we have two not initialize joints engine crash
        self.joint = self.input_type.motor()
        self.joint.Initialize(in_block.body, out_block.body, True,
                              in_block.transformed_frame_out, out_block.ref_frame_in)
        self.builder.AddLink(self.joint)

        if(self.stiffness != 0) or (self.damping != 0):
            self.add_spring_damper(in_block, out_block)

    def apply_transform(self, in_block):
        self.transformed_frame_out = self._ref_frame_out * in_block.transform

    def add_spring_damper(self, in_block: ChronoBody, out_block: ChronoBody):
        self.joint_spring = chrono.ChLinkRSDA()
        self.joint_spring.Initialize(in_block.body, out_block.body, False,
                                     in_block.transformed_frame_out.GetAbsCoord(),out_block.ref_frame_in.GetAbsCoord())
        self.torque_functor = SpringTorque(self.stiffness, self.damping, self.equilibrium_position)
        self.joint_spring.RegisterTorqueFunctor(self.torque_functor)
        self.builder.Add(self.joint_spring)


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
    for block in sequence[it:it + 2]:
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
                block.move_to_out_frame(previous_body_block)  # NOQA gryazuka
                if need_fix_joint:
                    block.make_fix_joint(previous_body_block)  # NOQA

                need_fix_joint = True
                previous_body_block = block

        elif block.block_type is BlockType.Bridge:
            need_fix_joint = False

        elif block.block_type is BlockType.Transform:
            sequence[it - 1].apply_transform(block)

    for it, block in enumerate(sequence):  # NOQA
        if block.block_type == BlockType.Bridge:

            block_in = find_body_from_two_previous_blocks(sequence, it)
            block_out = find_body_from_two_after_blocks(sequence, it)

            if block_in is None:
                raise Exception('Bridge block require body block before')
            if block_out is None:
                raise Exception('Bridge block require body block after')

            block.connect(block_in, block_out)  # NOQA
