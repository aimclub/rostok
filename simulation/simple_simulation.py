import random
from abc import ABC
from enum import Enum
from typing import Optional

import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
from functions import ContactReporter, SpringTorque, rotation_z_q, frame_transform_to_chcoordsys

from rostok.block_builder.transform_srtucture import FrameTransform
from rostok.utils.dataset_materials.material_dataclass_manipulating import (
    DefaultChronoMaterial, Material, struct_material2object_material)


class BlockType(str, Enum):
    TRANSFORM = "Transform"
    SELFTRANSFORM = "SelfTransform"
    BODY = "Body"
    BRIDGE = "Bridge"


class BlockBody(ABC):

    def __init__(self):
        self.block_type = BlockType.BODY
        self.body = None
        self.is_build = False


class ChronoBody(BlockBody, ABC):
    """Abstract class, that interpreting nodes of a robot body part in a
    physics engine (`pychrono <https://projectchrono.org/pychrono/>`_).
    
    Attributes:
        body (pychrono.ChBody): Pychrono object of the solid body. It defines visualisation,
        collision shape, position on the world frame and etc in simulation system.

    Args:
        builder (pychrono.ChSystem): Arg sets the system, which hosth the body
        body (pychrono.ChBody): Solid body define nodes of the body part in the physics system
        in_pos_marker (pychrono.ChVectorD): Arg defines position input frame the body
        out_pos_marker (chrono.ChVectorD): Arg defines position output frame the body
        random_color (bool): Flag of the random color of the body
        is_collide (bool, optional): Flag of collision body with other objects in system.
        Defaults to True.
    """

    def __init__(self,
                 body: chrono.ChBody,
                 in_pos_marker: chrono.ChVectorD,
                 out_pos_marker: chrono.ChVectorD,
                 color=None,
                 is_collide: bool = True):
        """Abstract class of interpretation of nodes of a robot body part in a
        physics engine.

        Initlization adds body in system, creates input and output
        marker of the body and sets them. Also, it initilize object of
        the contact reporter
        """
        super().__init__()
        self.body = body

        # Create markers - additional coordinate frames for the body
        input_marker = chrono.ChMarker()
        out_marker = chrono.ChMarker()
        transformed_out_marker = chrono.ChMarker()
        transformed_input_marker = chrono.ChMarker()

        input_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
        out_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
        transformed_out_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
        transformed_input_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)

        self.body.AddMarker(input_marker)
        self.body.AddMarker(out_marker)
        self.body.AddMarker(transformed_out_marker)
        self.body.AddMarker(transformed_input_marker)
        self.body.GetCollisionModel().SetDefaultSuggestedEnvelope(0.001)
        self.body.GetCollisionModel().SetDefaultSuggestedMargin(0.0005)
        self.body.SetCollide(is_collide)

        input_marker.SetPos(in_pos_marker)
        out_marker.SetPos(out_pos_marker)
        # Calc SetPos
        transformed_out_marker.SetCoord(out_marker.GetCoord())
        transformed_input_marker.SetCoord(input_marker.GetCoord())

        self._ref_frame_in = input_marker
        self._ref_frame_out = out_marker
        self.transformed_frame_out = transformed_out_marker
        self.transformed_frame_input = transformed_input_marker

        # Normal Forces
        self.__contact_reporter = ContactReporter(self.body)

        if color is None:
            rgb = [random.random(), random.random(), random.random()]
            rgb[int(random.random() * 2)] *= 0.2
            self.body.GetVisualShape(0).SetColor(chrono.ChColor(*rgb))
        else:
            color = [x / 256 for x in color]
            self.body.GetVisualShape(0).SetColor(chrono.ChColor(*color))

    def reset_transformed_frame_out(self):
        """Reset all transforms output frame of the body and back to initial
        state."""
        self.transformed_frame_out.SetCoord(self._ref_frame_out.GetCoord())

    def reset_transformed_frame_input(self):
        """Reset all transforms output frame of the body and back to initial
        state."""
        self.transformed_frame_input.SetCoord(self._ref_frame_in.GetCoord())

    def apply_transform(self, transform: chrono.ChCoordsysD):
        """Applied input transformation to the output frame of the body.

        Args:
            in_block (BlockTransform): The block which define transformations
        """
        self.reset_transformed_frame_out()
        frame_coord = self.transformed_frame_out.GetCoord()
        frame_coord = frame_coord * transform
        self.transformed_frame_out.SetCoord(frame_coord)

    def apply_input_transform(self, transform: chrono.ChCoordsysD):
        """Applied input transformation to the output frame of the body.

        Args:
            in_block (BlockTransform): The block which define transformations
        """
        self.reset_transformed_frame_input()
        frame_coord = self.transformed_frame_input.GetCoord()
        frame_coord = frame_coord * transform
        self.transformed_frame_input.SetCoord(frame_coord)

    def set_coord(self, transform: FrameTransform):
        self.body.SetCoord(frame_transform_to_chcoordsys(transform))

    @property
    def ref_frame_in(self) -> chrono.ChMarker:
        """Return the input frame of the body.

        Returns:
            pychrono.ChMarker: The input frame of the body
        """
        return self._ref_frame_in

    @property
    def normal_force(self, builder) -> float:
        """Return value normal forces of random collision point.

        Returns:
            float: Value normal forces of random collision point
        """
        builder.GetContactContainer().ReportAllContacts(self.__contact_reporter)
        return self.__contact_reporter.get_normal_forces()

    @property
    def list_n_forces(self, builder) -> list:
        """Return a list of all the contact forces.

        Returns:
            list: List normal forces of all the contacts points
        """
        container = builder.GetContactContainer()
        contacts = container.GetNcontacts()
        if contacts:
            self.__contact_reporter.list_clear()
            container.ReportAllContacts(self.__contact_reporter)
        return self.__contact_reporter.get_list_n_forces()

    @property
    def list_c_coord(self, builder) -> list:
        """Return a list of all the contact forces.
        Returns:
            list: List normal forces of all the contacts points
        """
        container = builder.GetContactContainer()
        contacts = container.GetNcontacts()
        if contacts:
            self.__contact_reporter.list_cont_clear()
            container.ReportAllContacts(self.__contact_reporter)
        return self.__contact_reporter.get_list_c_coord()


# A class to build an EasyBox that have functionality of ChronoBody
class UniversalBox(ChronoBody):

    def __init__(self,
                 x,
                 y,
                 z,
                 color: list[int] = [256, 256, 256],
                 material: Material = DefaultChronoMaterial(),
                 is_collide: bool = True,
                 pos: FrameTransform = FrameTransform([0, 0.0, 0], [1, 0, 0, 0])):

        MOCK_DENSITY: int = 10
        material = struct_material2object_material(material)
        body = chrono.ChBodyEasyBox(x, y, z, MOCK_DENSITY, True, True, material)
        body.SetCoord(frame_transform_to_chcoordsys(pos))
        body.GetCollisionModel().SetDefaultSuggestedEnvelope(0.001)
        body.GetCollisionModel().SetDefaultSuggestedMargin(0.0005)
        pos_in_marker = chrono.ChVectorD(0, -y * 0.5, 0)
        pos_out_marker = chrono.ChVectorD(0, y * 0.5, 0)
        super().__init__(body, pos_in_marker, pos_out_marker, color)


class BlockTransform(ABC):

    def __init__(self):
        self.block_type = BlockType.TRANSFORM


class BlockBridge(ABC):

    def __init__(self):
        self.block_type = BlockType.BRIDGE


class ChronoTransform(BlockTransform):
    """Class representing node of the transformation in `pychrono <https://projectchrono.org/pychrono/>`_ physical
    engine

    Args:
        builder (pychrono.ChSystem): Arg sets the system, which hosth the body
        transform (FrameTransform): Define tranformation of the instance
    """

    def __init__(self, transform):
        super().__init__()
        if isinstance(transform, chrono.ChCoordsysD):
            self.transform = transform
        elif isinstance(transform, FrameTransform):
            coordsys_transform = chrono.ChCoordsysD(
                chrono.ChVectorD(transform.position[0], transform.position[1],
                                 transform.position[2]),
                chrono.ChQuaternionD(transform.rotation[0], transform.rotation[1],
                                     transform.rotation[2], transform.rotation[3]))
            self.transform = coordsys_transform


class BlockSelfTransform(ABC):

    def __init__(self):
        self.block_type = BlockType.SELFTRANSFORM


class ChronoSelfTransform(BlockSelfTransform):
    """Class representing node of the transformation in `pychrono <https://projectchrono.org/pychrono/>`_ physical
    engine

    Args:
        builder (pychrono.ChSystem): Arg sets the system, which hosth the body
        transform (FrameTransform): Define tranformation of the instance
    """

    def __init__(self, transform):
        super().__init__()
        if isinstance(transform, chrono.ChCoordsysD):
            self.transform = transform
        elif isinstance(transform, FrameTransform):
            coordsys_transform = chrono.ChCoordsysD(
                chrono.ChVectorD(transform.position[0], transform.position[1],
                                 transform.position[2]),
                chrono.ChQuaternionD(transform.rotation[0], transform.rotation[1],
                                     transform.rotation[2], transform.rotation[3]))
            self.transform = coordsys_transform


class ChronoRevolveJoint(BlockBridge):
    """The class representing revolute joint object in `pychrono <https://projectchrono.org/pychrono/>`_ physical
    engine. It is the embodiment of joint nodes from the mechanism graph in
    simulation.

        Args:
            builder (pychrono.ChSystem): Arg sets the system, which hosth the body
            axis (Axis, optional): Define rotation axis. Defaults to Axis.Z.
            type_of_input (InputType, optional): Define type of input joint control. Defaults to InputType.POSITION. Instead of, can changes to torque, that more realistic.
            stiffness (float, optional): Optional arg add a spring with `stiffness` to joint. Defaults to 0.
            damping (float, optional): Optional arg add a dempher to joint. Defaults to 0.
            equilibrium_position (float, optional): Define equilibrium position of the spring. Defaults to 0.

        Attributes:
            joint (pychrono.ChLink): Joint define nodes of the joint part in the system
            axis (Axis): The axis of the rotation
            input_type (InputType): The type of input
    """

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
                 axis: Axis = Axis.Z,
                 type_of_input: InputType = InputType.TORQUE,
                 radius=0.05,
                 length = 0.4,
                 starting_angle=0,
                 stiffness: float = 0.,
                 damping: float = 0.,
                 equilibrium_position: float = 0.):
        super().__init__()
        self.joint = None
        self.axis = axis
        self.input_type = type_of_input
        self.radius = radius
        self.length = length
        self.starting_angle = starting_angle
        # Spring Damper params
        self._joint_spring = None
        self._torque_functor = None
        self.stiffness = stiffness
        self.damping = damping
        self.equilibrium_position = equilibrium_position

    def set_prev_body_frame(self, prev_block: ChronoBody, system: chrono.ChSystem):
        # additional transform is just a translatoin along y axis to the radius of the joint
        additional_transform = chrono.ChCoordsysD(chrono.ChVectorD(0, self.radius, 0),
                                                  chrono.ChQuaternionD(1, 0, 0, 0))
        additional_transform *= chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0),
                                                   rotation_z_q(self.starting_angle))
        transform = prev_block.transformed_frame_out.GetCoord()
        prev_block.transformed_frame_out.SetCoord(transform * additional_transform)
        system.Update()

    def set_next_body_frame(self, next_block: ChronoBody, system: chrono.ChSystem):
        additional_transform = chrono.ChCoordsysD(chrono.ChVectorD(0, -self.radius, 0),
                                                  chrono.ChQuaternionD(1, 0, 0, 0))
        transform = next_block.transformed_frame_input.GetCoord()
        next_block.transformed_frame_input.SetCoord(transform * additional_transform)
        system.Update()

    def connect(self, in_block: ChronoBody, out_block: ChronoBody, system: chrono.ChSystem):
        """Joint is connected two bodies.

        If we have two not initialize joints engine crash

        Args:
            in_block (ChronoBody): Slave body to connect
            out_block (ChronoBody): Master body to connect
        """
        system.Update()
        self.joint:chrono.ChLinkMotorRotationTorque = self.input_type.motor()
                # Add cylinder visual
        cylinder = chrono.ChCylinder()
        cylinder.p2 = chrono.ChVectorD(0, 0,
                                       self.length / 2)
        cylinder.p1 = chrono.ChVectorD(0, 0,
                                       -self.length / 2)
        cylinder.rad = self.radius
        cylinder_asset = chrono.ChCylinderShape(cylinder)
        self.joint.AddVisualShape(cylinder_asset)
        #self.joint.AddCollisionModelsToSystem
        self.joint.Initialize(in_block.body, out_block.body, True, in_block.transformed_frame_out,
                              out_block.transformed_frame_input)
        system.AddLink(self.joint)

        if (self.stiffness != 0) or (self.damping != 0):
            self._add_spring_damper(in_block, out_block, system)
        system.Update()

    def _add_spring_damper(self, in_block: ChronoBody, out_block: ChronoBody,
                           system: chrono.ChSystem):
        self._joint_spring = chrono.ChLinkRSDA()
        self._joint_spring.Initialize(in_block.body, out_block.body, False,
                                      in_block.transformed_frame_out.GetAbsCoord(),
                                      out_block.ref_frame_in.GetAbsCoord())
        self._torque_functor = SpringTorque(self.stiffness, self.damping, self.equilibrium_position)
        self._joint_spring.RegisterTorqueFunctor(self._torque_functor)
        system.Add(self._joint_spring)


def place_next_block(prev_block:ChronoBody, next_block:ChronoBody, system: chrono.ChSystem):
    # prev_body is already added to the system
    prev_body: chrono.ChBody = prev_block.body
    next_body: chrono.ChBody = next_block.body
    total_transformation = prev_block.transformed_frame_out.GetAbsCoord() * chrono.ChFrameD(
        next_block.transformed_frame_input).GetInverse().GetCoord()
    print(prev_block.transformed_frame_out.GetAbsCoord().pos, prev_block.transformed_frame_out.GetAbsCoord().rot)
    print(chrono.ChFrameD(next_block.transformed_frame_input).GetInverse().GetCoord().pos)
    print(total_transformation.pos,total_transformation.rot)
    print()
    next_body.SetCoord(total_transformation)
    system.Add(next_body)
    system.Update()


def make_fix_joint(prev_block, next_block, system: chrono.ChSystem):

    prev_body = prev_block.body
    next_body = next_block.body

    fix_joint = chrono.ChLinkMateFix()
    fix_joint.Initialize(prev_body, next_body, True, prev_block.transformed_frame_out,
                         next_block.transformed_frame_input)
    system.Add(fix_joint)
    system.Update()


def start_selftransformation_loop():
    pass


# the function places and connects a sequence of blocks. The sequence should start from the root block
def place_and_connect(sequence: list, system: chrono.ChSystem):
    # all connections occurs between bodies
    previous_body_block = None
    previous_joint = None
    for it, block in enumerate(sequence):
        if block.block_type is BlockType.BODY:
            if previous_body_block is None:
                # the body is first in sequence
                previous_body_block = block
                if not block.is_build:
                    system.AddBody(block.body)
                    system.Update()

            else:
                if not block.is_build:
                    if sequence[it - 1].block_type is BlockType.SELFTRANSFORM:
                        start_selftransformation_loop()

                    if previous_joint is None:
                        place_next_block(previous_body_block, block, system)
                        make_fix_joint(previous_body_block, block, system)
                    else:
                        previous_joint.set_next_body_frame(block, system)
                        place_next_block(previous_body_block, block, system)
                        previous_joint.connect(previous_body_block, block, system)
                        previous_joint = None


                previous_body_block = block

            block.is_build = True
        # transforms follow the block and shift the outer transformed frame,
        # using inner function `apply_transform`
        elif block.block_type is BlockType.TRANSFORM:
            i = 0
            transform = True
            while transform:
                i += 1
                if sequence[it - i].block_type is BlockType.BODY:
                    transform = False
                    current_transform = chrono.ChCoordsysD()
                    for k in range(it - i + 1, it + 1):
                        current_transform = current_transform * sequence[k].transform

                    sequence[it - i].apply_transform(current_transform)
                elif sequence[it - i].block_type is BlockType.BRIDGE:
                    raise Exception("Transform after joint!!!")
                else:
                    continue
        # self transformations follows are applied to the next body input frame
        elif block.block_type is BlockType.SELFTRANSFORM:
            continue

        elif block.block_type is BlockType.BRIDGE:
            block.set_prev_body_frame(previous_body_block, system)
            previous_joint = block


if __name__ == "__main__":
    chrono_system = chrono.ChSystemNSC()
    chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    chrono_system.SetSolverMaxIterations(100)
    chrono_system.SetSolverForceTolerance(1e-6)
    chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    chrono_system.Set_G_acc(chrono.ChVectorD(0, 0, 0))
    flat = UniversalBox(1, 0.2, 1)
    joint = ChronoRevolveJoint(starting_angle=45)
    
    link = UniversalBox(0.1, 0.6, 0.4)

    block_list = [flat, joint, link]

    place_and_connect(block_list, chrono_system)
    flat.body.SetBodyFixed(True)
    joint.joint.SetTorqueFunction(chrono.ChFunction_Const(1))
    coord_flat = flat.transformed_frame_out.GetCoord()
    coord_link = link.transformed_frame_input.GetCoord()
    coord_link_abs = link.transformed_frame_input.GetAbsCoord()

    print(coord_flat.pos, coord_flat.rot)
    print(coord_link.pos, coord_link.rot)
    print(coord_link_abs.pos, coord_link_abs.rot)


    vis = chronoirr.ChVisualSystemIrrlicht()
    vis.AttachSystem(chrono_system)
    vis.SetWindowSize(1024, 768)
    vis.SetWindowTitle('Grab demo')
    vis.Initialize()
    vis.AddCamera(chrono.ChVectorD(1.5, 3, -2))
    vis.AddTypicalLights()
    vis.EnableCollisionShapeDrawing(True)

    # Simulation loop
    while vis.Run():
        chrono_system.Update()
        chrono_system.DoStepDynamics(5e-3)
        vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
        vis.Render()
        vis.EndScene()
    # place_and_connect(body_list)
