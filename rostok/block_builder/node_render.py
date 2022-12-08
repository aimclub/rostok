from enum import Enum
from abc import ABC
from typing import Optional
import random

import pychrono.cascade as cascade
import pychrono.core as chrono

from OCC.Core import BRepPrimAPI
from OCC.Core import BRepAlgoAPI
from OCC.Core import gp

from rostok.utils.dataset_materials.material_dataclass_manipulating import (
    DefaultChronoMaterial, struct_material2object_material)
from rostok.block_builder.transform_srtucture import FrameTransform
from rostok.block_builder.basic_node_block import (BlockBody, RobotBody, BlockBridge, BlockType,
                                                   Block, BlockTransform, SimpleBody)


class SpringTorque(chrono.TorqueFunctor):

    def __init__(self, spring_coef, damping_coef, rest_angle):
        super(SpringTorque, self).__init__()
        self.spring_coef = spring_coef
        self.damping_coef = damping_coef
        self.rest_angle = rest_angle

    def evaluate(
            self,  #
            time,  # current time
            angle,  # relative angle of rotation
            vel,  # relative angular speed
            link):  # back-pointer to associated link
        torque = 0
        if self.spring_coef > 10**-3:
            torque = -self.spring_coef * \
                (angle - self.rest_angle) - self.damping_coef * vel
        else:
            torque = -self.damping_coef * vel
        return torque


class ContactReporter(chrono.ReportContactCallback):

    def __init__(self, chrono_body):
        """Create a reporter normal contact forces for the body

        Args:
            chrono_body (ChBody): Repoter's body
        """
        self._body = chrono_body
        self.__current_normal_forces = None
        self.__list_normal_forces = []
        super().__init__()

    def OnReportContact(
        self,
        pA,  # contact pA 
        pB,  # contact pB 
        plane_coord,  # contact plane coordsystem (A column 'X' is contact normal) 
        distance,  # contact distance 
        eff_radius,  # effective radius of curvature at contact 
        cforce,  # react.forces (if already computed). In coordsystem 'plane_coord' 
        ctorque,  # react.torques, if rolling friction (if already computed). 
        modA,  # model A (note: some containers may not support it and could be nullptr) 
        modB):  # model B (note: some containers may not support it and could be nullptr)
        bodyA = chrono.CastToChBody(modA)
        bodyB = chrono.CastToChBody(modB)
        if (bodyA == self._body) or (bodyB == self._body):
            self.__current_normal_forces = cforce.x
            self.__list_normal_forces.append(cforce.x)
        return True

    def is_empty(self):
        return len(self.__list_normal_forces) == 0

    def list_clear(self):
        self.__list_normal_forces.clear()

    def get_normal_forces(self):
        return self.__current_normal_forces

    def get_list_n_forces(self):
        return self.__list_normal_forces


class ChronoBody(BlockBody, ABC):

    def __init__(self, builder: chrono.ChSystem, body: chrono.ChBody, in_pos_marker: chrono.ChVectorD,
                 out_pos_marker: chrono.ChVectorD, random_color: bool):
        super().__init__(builder)
        self.body = body
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
        self.body.GetCollisionModel().SetDefaultSuggestedEnvelope(0.001)
        self.body.GetCollisionModel().SetDefaultSuggestedMargin(0.0005)

        input_marker.SetPos(in_pos_marker)
        out_marker.SetPos(out_pos_marker)

        # Calc SetPos
        transformed_out_marker.SetCoord(out_marker.GetCoord())

        self._ref_frame_in = input_marker
        self._ref_frame_out = out_marker
        self.transformed_frame_out = transformed_out_marker

        # Normal Forces
        self.__contact_reporter = ContactReporter(self.body)

        if random_color:
            rgb = [random.random(), random.random(), random.random()]
            rgb[int(random.random() * 2)] *= 0.2
            self.body.GetVisualShape(0).SetColor(chrono.ChColor(*rgb))

    def _build_collision_model(self, struct_material, width, length):
        """Build collision model of the block on material width and length

        Args:
            struct_material (Material): Dataclass of material body
            width (flaot): Width of the box
            length (float): Length of the box
        """
        chrono_object_material = struct_material2object_material(struct_material)

        self.body.GetCollisionModel().ClearModel()
        self.body.GetCollisionModel().AddBox(chrono_object_material, width / 2, length / 2,
                                             width / 2)
        self.body.GetCollisionModel().BuildModel()

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

    @property
    def normal_force(self):
        self.builder.GetContactContainer().ReportAllContacts(self.__contact_reporter)
        return self.__contact_reporter.get_normal_forces()

    @property
    def list_n_forces(self):
        container = self.builder.GetContactContainer()
        contacts = container.GetNcontacts()
        if contacts:
            self.__contact_reporter.list_clear()
            container.ReportAllContacts(self.__contact_reporter)
        return self.__contact_reporter.get_list_n_forces()


class BasicChronoBody(ChronoBody, RobotBody):

    def __init__(self,
                 builder,
                 length=2,
                 width=0.1,
                 depth=0.3,
                 random_color=True,
                 mass=1,
                 material=DefaultChronoMaterial()):

        # Create body
        material = struct_material2object_material(material)
        box_s = BRepPrimAPI.BRepPrimAPI_MakeBox(gp.gp_Pnt(-width / 2, 0, 0), width, length,
                                                depth).Shape()
        cylinder_s = BRepPrimAPI.BRepPrimAPI_MakeCylinder(width / 2, depth).Shape()
        shape = BRepAlgoAPI.BRepAlgoAPI_Fuse(box_s, cylinder_s).Shape()
        vis_params = cascade.ChCascadeTriangulate(1, True, 0.5)
        body = cascade.ChCascadeBodyEasy(shape, 1000, vis_params, True, material)
        body.SetMass(mass)

        # Create shape
        # TODO: setter for shape
        pos_in_marker = chrono.ChVectorD(0, -length / 2, 0)
        pos_out_marker = chrono.ChVectorD(0, length / 2 + width / 2 + 0.5 * width, 0)
        super().__init__(builder, body, pos_in_marker, pos_out_marker, random_color)


class FlatChronoBody(ChronoBody, RobotBody):

    def __init__(self,
                 builder,
                 length=2,
                 width=0.1,
                 depth=0.3,
                 random_color=True,
                 mass=1,
                 material=DefaultChronoMaterial()):
        # Create body

        body = chrono.ChBody()

        box_asset = chrono.ChBoxShape()
        box_asset.GetBoxGeometry().Size = chrono.ChVectorD(width / 2, length / 2, depth / 2)
        body.AddVisualShape(box_asset)
        body.SetCollide(True)

        body.SetMass(mass)

        pos_input_marker = chrono.ChVectorD(0, -length / 2, 0)
        pos_out_marker = chrono.ChVectorD(0, length / 2 + width / 16, 0)
        super().__init__(builder, body, pos_input_marker, pos_out_marker, random_color)

        self._build_collision_model(material, width, length)
        # Create shape
        # TODO: setter for shape


class MountChronoBody(ChronoBody, RobotBody):

    def __init__(self,
                 builder,
                 length=2,
                 width=0.1,
                 depth=0.3,
                 random_color=True,
                 mass=1,
                 material=DefaultChronoMaterial()):
        # Create body

        body = chrono.ChBody()

        box_asset = chrono.ChBoxShape()
        box_asset.GetBoxGeometry().Size = chrono.ChVectorD(width / 2, length / 2, depth / 2)
        body.AddVisualShape(box_asset)
        body.SetCollide(True)

        body.SetMass(mass)

        pos_input_marker = chrono.ChVectorD(0, -length / 2 + width / 2, 0)
        pos_out_marker = chrono.ChVectorD(0, length / 2, 0)
        super().__init__(builder, body, pos_input_marker, pos_out_marker, random_color)

        self._build_collision_model(material, width, length)


class ChronoBodyEnv(ChronoBody):

    def __init__(self,
                 builder,
                 shape=SimpleBody.BOX,
                 random_color=True,
                 mass=1,
                 material=DefaultChronoMaterial(),
                 pos: FrameTransform = FrameTransform([0, 0.0, 0], [1, 0, 0, 0])):
        
        # Create body
        material = struct_material2object_material(material)
        if shape.name == "BOX":
            body = chrono.ChBodyEasyBox(shape.value.width, shape.value.length, shape.value.height,
                                        1000, True, True, material)
        elif shape.name == "CYLINDER":
            body = chrono.ChBodyEasyCylinder(shape.value.radius, shape.value.height, 1000, True, True,
                                             material)
        elif shape.name == "SPHERE":
            body = chrono.ChBodyEasySphere(shape.value.radius, 1000, True, True, material)
        elif shape.name == "ELLIPSOID":
            body = chrono.ChBodyEasyEllipsoid(
                chrono.ChVectorD(shape.value.radius_a, shape.value.radius_b, shape.value.radius_c),
                1000, True, True, material)
        body.SetCollide(True)
        transform = ChronoTransform(builder, pos)
        body.SetCoord(transform.transform)

        body.SetMass(mass)

        # Create shape
        pos_in_marker = chrono.ChVectorD(0, 0, 0)
        pos_out_marker = chrono.ChVectorD(0, 0, 0)
        super().__init__(builder, body, pos_in_marker, pos_out_marker, random_color)

    def set_coord(self, frame: FrameTransform):
        transform = ChronoTransform(self.builder, frame)
        self.body.SetCoord(transform.transform)


class ChronoRevolveJoint(BlockBridge):
    # Variants of joint control
    class InputType(str, Enum):
        TORQUE = {"Name": "Torque", "TypeMotor": chrono.ChLinkMotorRotationTorque}
        VELOCITY = {"Name": "Speed", "TypeMotor": chrono.ChLinkMotorRotationSpeed}
        POSITION = {"Name": "Angle", "TypeMotor": chrono.ChLinkMotorRotationAngle}
        UNCONTROL = {"Name": "Uncontrol", "TypeMotor": chrono.ChLinkRevolute}

        def __init__(self, vals):
            self.num = vals["Name"]
            self.motor = vals["TypeMotor"]

    class Axis(str, Enum):
        # Z is default rotation axis
        Z = chrono.ChQuaternionD(1, 0, 0, 0)
        Y = chrono.Q_ROTATE_Z_TO_Y
        X = chrono.Q_ROTATE_Z_TO_X

    def __init__(self,
                 builder,
                 axis=Axis.Z,
                 type_of_input=InputType.POSITION,
                 stiffness=0,
                 damping=0,
                 equilibrium_position=0):
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
        self.joint.Initialize(in_block.body, out_block.body, True, in_block.transformed_frame_out,
                              out_block.ref_frame_in)
        self.builder.AddLink(self.joint)

        if (self.stiffness != 0) or (self.damping != 0):
            self.add_spring_damper(in_block, out_block)

    def apply_transform(self, in_block):
        self.transformed_frame_out = self._ref_frame_out * in_block.transform

    def add_spring_damper(self, in_block: ChronoBody, out_block: ChronoBody):
        self.joint_spring = chrono.ChLinkRSDA()
        self.joint_spring.Initialize(in_block.body, out_block.body, False,
                                     in_block.transformed_frame_out.GetAbsCoord(),
                                     out_block.ref_frame_in.GetAbsCoord())
        self.torque_functor = SpringTorque(self.stiffness, self.damping, self.equilibrium_position)
        self.joint_spring.RegisterTorqueFunctor(self.torque_functor)
        self.builder.Add(self.joint_spring)


class ChronoTransform(BlockTransform):

    def __init__(self, builder: chrono.ChSystem, transform):
        super().__init__(builder=builder)
        if isinstance(transform, chrono.ChCoordsysD):
            self.transform = transform
        elif type(transform) is FrameTransform:
            coordsys_transform = chrono.ChCoordsysD(
                chrono.ChVectorD(transform.position[0], transform.position[1],
                                 transform.position[2]),
                chrono.ChQuaternionD(transform.rotation[0], transform.rotation[1],
                                     transform.rotation[2], transform.rotation[3]))
            self.transform = coordsys_transform


def find_body_from_two_previous_blocks(sequence: list[Block], it: int) -> Optional[Block]:
    # b->t->j->t->b Longest sequence
    for i in reversed(range(it)[-2:]):
        if sequence[i].block_type == BlockType.BODY:
            return sequence[i]
    return None


def find_body_from_two_after_blocks(sequence: list[Block], it: int) -> Optional[Block]:
    # b->t->j->t->b Longest sequence
    for block in sequence[it:it + 2]:
        if block.block_type == BlockType.BODY:
            return block
    return None


def connect_blocks(sequence: list[Block]):
    # Make body and apply transform
    previous_body_block = None
    need_fix_joint = False

    for it, block in enumerate(sequence):
        if block.block_type is BlockType.BODY:
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

        elif block.block_type is BlockType.BRIDGE:
            need_fix_joint = False

        elif block.block_type is BlockType.TRANSFORM:
            sequence[it - 1].apply_transform(block)

    for it, block in enumerate(sequence):  # NOQA
        if block.block_type == BlockType.BRIDGE:

            block_in = find_body_from_two_previous_blocks(sequence, it)
            block_out = find_body_from_two_after_blocks(sequence, it)

            if block_in is None:
                raise Exception('Bridge block require body block before')
            if block_out is None:
                raise Exception('Bridge block require body block after')

            block.connect(block_in, block_out)  # NOQA
