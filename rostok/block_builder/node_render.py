from enum import Enum
from abc import ABC
from typing import Optional
import random

import pychrono.core as chrono

from rostok.block_builder.body_size import BoxSize

from rostok.utils.dataset_materials.material_dataclass_manipulating import (
    DefaultChronoMaterial, Material, struct_material2object_material)
from rostok.block_builder.transform_srtucture import FrameTransform
from rostok.block_builder.basic_node_block import (BlockBody, RobotBody, BlockBridge, BlockType,
                                                   Block, BlockTransform, SimpleBody)


#Constants

ENVELOPE = 0.001
MARGIN = 0.0005

class SpringTorque(chrono.TorqueFunctor):

    def __init__(self, spring_coef, damping_coef, rest_angle):
        super(SpringTorque, self).__init__()
        self.spring_coef = spring_coef
        self.damping_coef = damping_coef
        self.rest_angle = rest_angle

    def evaluate(self, time, angle, vel, link):
        """Calculation of torque, that is created by spring
        

        Args:
            time  :  current time
            angle :  relative angle of rotation
            vel   :  relative angular speed
            link  :  back-pointer to associated link


        Returns:
            torque: torque, that is created by spring
        """
        torque = 0
        if self.spring_coef > 10**-3:
            torque = -self.spring_coef * \
                (angle - self.rest_angle) - self.damping_coef * vel
        else:
            torque = -self.damping_coef * vel
        return torque


class ContactReporter(chrono.ReportContactCallback):

    def __init__(self, chrono_body):
        """Create a sensor of contact normal forces for the body.

        Args:
            chrono_body (ChBody): The body on which the sensor is install
        """
        self._body = chrono_body
        self.__current_normal_forces = None
        self.__current_contact_coord = None
        self.__list_normal_forces = []
        self.__list_contact_coord = []
        super().__init__()


    def OnReportContact(self, pA: chrono.ChVectorD, pB: chrono.ChVectorD,
                        plane_coord: chrono.ChMatrix33D, distance: float, eff_radius: float,
                        react_forces: chrono.ChVectorD, react_torques: chrono.ChVectorD,
                        contactobjA: chrono.ChContactable, contactobjB: chrono.ChContactable):
        """Callback used to report contact points already added to the container

        Args:
            pA (ChVector): coordinates of contact point(s) in body A
            pB (ChVector): coordinates of contact point(s) in body B
            plane_coord (ChMatrix33): contact plane coordsystem
            distance (float): contact distance
            eff_radius (float)): effective radius of curvature at contact
            react_forces (ChVector): reaction forces in coordsystem 'plane_coord'
            react_torques (ChVector): reaction torques, if rolling friction
            contactobjA (ChContactable): model A
            contactobjB (ChContactable): model B
        Returns:
            bool: If returns false, the contact scanning will be stopped
        """

        body_a = chrono.CastToChBody(contactobjA)
        body_b = chrono.CastToChBody(contactobjB)
        if (body_a == self._body) or (body_b == self._body):
            self.__current_normal_forces = react_forces.x
            self.__list_normal_forces.append(react_forces.x)

            if (body_a == self._body):
                self.__current_contact_coord = [pA.x, pA.y, pA.z]
                self.__list_contact_coord.append(self.__current_contact_coord)
            elif(body_b == self._body):
                self.__current_contact_coord = [pB.x, pB.y, pB.z]
                self.__list_contact_coord.append(self.__current_contact_coord)

        return True

    def is_empty(self):
        return len(self.__list_normal_forces) == 0

    def list_clear(self):
        self.__list_normal_forces.clear()

    def list_cont_clear(self):
        self.__list_contact_coord.clear()

    def get_normal_forces(self):
        return self.__current_normal_forces

    def get_list_n_forces(self):
        return self.__list_normal_forces

    def get_list_c_coord(self):
        return self.__list_contact_coord


class ChronoBody(BlockBody, ABC):
    """Abstract class, that interpreting nodes of a robot body part in a
    physics engine (`pychrono <https://projectchrono.org/pychrono/>`_).
    
    Attributes:
        body (pychrono.ChBody): Pychrono object of the solid body. It defines visualisation,
        collision shape, position on the world frame and etc in simulation system.
        builder (pychrono.ChSystem): Pychrono object of system, which hosts the body.

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
                 builder: chrono.ChSystem,
                 body: chrono.ChBody,
                 in_pos_marker: chrono.ChVectorD,
                 out_pos_marker: chrono.ChVectorD,
                 random_color: bool,
                 is_collide: bool = True):
        """Abstract class of interpretation of nodes of a robot body part in a
        physics engine.

        Initlization adds body in system, creates input and output
        marker of the body and sets them. Also, it initilize object of
        the contact reporter
        """
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
        self.body.GetCollisionModel().SetDefaultSuggestedEnvelope(ENVELOPE)
        self.body.GetCollisionModel().SetDefaultSuggestedMargin(MARGIN)
        self.body.SetCollide(is_collide)

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

    def _build_collision_box_model(self, struct_material, width, length):
        """Build collision model of the block on material width and length.

        Args:
            struct_material (Material): Dataclass of material body
            width (float): Width of the box
            length (float): Length of the box
        """
        chrono_object_material = struct_material2object_material(struct_material)

        self.body.GetCollisionModel().ClearModel()
        self.body.GetCollisionModel().AddBox(chrono_object_material, width / 2, length / 2,
                                             width / 2)
        self.body.GetCollisionModel().BuildModel()

    def move_to_out_frame(self, in_block: Block):
        """Move the input frame body to output frame position input block.

        Args:
            in_block (Block): The block defines relative movming to output frame
        """
        self.builder.Update()
        local_coord_in_frame = self._ref_frame_in.GetCoord()
        abs_coord_out_frame = in_block.transformed_frame_out.GetAbsCoord()

        trans = chrono.ChFrameD(local_coord_in_frame)
        trans = trans.GetInverse()
        trans = trans.GetCoord()
        coord = abs_coord_out_frame * trans

        self.body.SetCoord(coord)

    def make_fix_joint(self, in_block):
        """Create weld joint (fixing relative posiotion and orientation)
        between input block and the body.

        Args:
            in_block (Block): The block which define relative fixing position and orientation the
            body in system
        """
        fix_joint = chrono.ChLinkMateFix()
        fix_joint.Initialize(in_block.body, self.body)
        self.builder.Add(fix_joint)

    def reset_transformed_frame_out(self):
        """Reset all transforms output frame of the body and back to initial
        state."""
        self.transformed_frame_out.SetCoord(self._ref_frame_out.GetCoord())

    def apply_transform(self, in_block: BlockTransform):
        """Applied input transformation to the output frame of the body.

        Args:
            in_block (BlockTransform): The block which define transformations
        """
        self.reset_transformed_frame_out()
        frame_coord = self.transformed_frame_out.GetCoord()
        frame_coord = frame_coord * in_block.transform
        self.transformed_frame_out.SetCoord(frame_coord)

    @property
    def ref_frame_in(self) -> chrono.ChMarker:
        """Return the input frame of the body.

        Returns:
            pychrono.ChMarker: The input frame of the body
        """
        return self._ref_frame_in

    @property
    def normal_force(self) -> float:
        """Return value normal forces of random collision point.

        Returns:
            float: Value normal forces of random collision point
        """
        self.builder.GetContactContainer().ReportAllContacts(self.__contact_reporter)
        return self.__contact_reporter.get_normal_forces()

    @property
    def list_n_forces(self) -> list:
        """Return a list of all the contact forces.

        Returns:
            list: List normal forces of all the contacts points
        """
        container = self.builder.GetContactContainer()
        contacts = container.GetNcontacts()
        if contacts:
            self.__contact_reporter.list_clear()
            container.ReportAllContacts(self.__contact_reporter)
        return self.__contact_reporter.get_list_n_forces()

    @property
    def list_c_coord(self) -> list:
        """Return a list of all the contact forces.
        Returns:
            list: List normal forces of all the contacts points
        """
        container = self.builder.GetContactContainer()
        contacts = container.GetNcontacts()
        if contacts:
            self.__contact_reporter.list_cont_clear()
            container.ReportAllContacts(self.__contact_reporter)
        return self.__contact_reporter.get_list_c_coord()


class BoxChronoBody(ChronoBody, RobotBody):
    """Class of the simple box body shape of robot on pychrono engine. It
    defines interpretation of node of body part in physic system `pychrono <https://projectchrono.org/pychrono/>`_
    
    Args:
        builder (chrono.ChSystem): Arg sets the system, which hosth the body
        size (BoxSize, optional): Size of the body box. Defaults to BoxSize(0.1, 0.1, 0.1).
        random_color (bool, optional): Flag of the random color of the body. Defaults to True.
        mass (float, optional): Value mass of the body box. Defaults to 1.
        material (Material, optional): Surface material, which define contact friction and etc.
        Defaults to DefaultChronoMaterial.
        is_collide (bool, optional): Flag of collision body with othe object in system.
        Defaults to True.
    """

    def __init__(self,
                 builder: chrono.ChSystem,
                 size: BoxSize = BoxSize(0.1, 0.1, 0.1),
                 random_color: bool = True,
                 mass: float = 1,
                 material: Material = DefaultChronoMaterial(),
                 is_collide: bool = True):
        # Create body
        material = struct_material2object_material(material)
        body = chrono.ChBody()

        box_asset = chrono.ChBoxShape()
        box_asset.GetBoxGeometry().Size = chrono.ChVectorD(size.width / 2, size.length / 2,
                                                           size.height / 2)
        body.AddVisualShape(box_asset)

        body.SetMass(mass)

        pos_in_marker = chrono.ChVectorD(0, -size.length / 2, 0)
        pos_out_marker = chrono.ChVectorD(0, size.length / 2, 0)
        super().__init__(builder,
                         body,
                         pos_in_marker,
                         pos_out_marker,
                         random_color,
                         is_collide=is_collide)
        self._build_collision_box_model(material, size.width, size.length)


class LinkChronoBody(ChronoBody, RobotBody):
    """Class interpretation of node of the link robot in physic engine
    `pychrono <https://projectchrono.org/pychrono/>`_.
    
    Args:
        builder (chrono.ChSystem): Arg sets the system, which hosting the body
        length (float): Length of the robot link. Defaults to 2.
        width (float): Width of the robot link. Defaults to 0.1.
        depth (float): Height of the robot link. Defaults to 0.3.
        random_color (bool, optional): Flag of the random color of the body. Defaults to True.
        mass (float, optional): Value mass of the body box. Defaults to 1.
        material (Material, optional): Surface material, which define contact friction and etc.
        Defaults to DefaultChronoMaterial.
        is_collide (bool, optional): Flag of collision body with other object in system.
        Defaults to True.
    """

    def __init__(self,
                 builder: chrono.ChSystem,
                 length: float = 2,
                 width: float = 0.1,
                 depth: float = 0.3,
                 random_color: bool = True,
                 mass: float = 1,
                 material: Material = DefaultChronoMaterial(),
                 is_collide: bool = True):

        # Create body
        material = struct_material2object_material(material)
        body = chrono.ChBody()

        # Calculate new length with gap
        gap_between_bodies = 0.05
        cylinder_r = width / 2
        offset = gap_between_bodies + cylinder_r
        length_minus_gap = length - offset

        if (length_minus_gap < 0):
            raise Exception(
                f"Soo short link length: {length} Need: length > width / 2 + {gap_between_bodies}")

        # Add box visual
        box_asset = chrono.ChBoxShape()
        #TODO: Move box asset + gap + cylinder_r
        box_asset.GetBoxGeometry().Size = chrono.ChVectorD(width / 2, (length - 2 * offset) / 2,
                                                           depth / 2)

        body.AddVisualShape(box_asset)

        # Add cylinder visual
        cylinder = chrono.ChCylinder()
        cylinder.p2 = chrono.ChVectorD(0, -length / 2 + gap_between_bodies + cylinder_r, depth / 2)
        cylinder.p1 = chrono.ChVectorD(0, -length / 2 + gap_between_bodies + cylinder_r, -depth / 2)
        cylinder.rad = cylinder_r
        cylinder_asset = chrono.ChCylinderShape(cylinder)
        body.AddVisualShape(cylinder_asset)

        # Add collision box
        body.GetCollisionModel().ClearModel()
        body.GetCollisionModel().AddBox(
            material, width / 2, length_minus_gap / 2, depth / 2,
            chrono.ChVectorD(0, (cylinder_r + gap_between_bodies) / 2, 0))

        # Add collision cylinder
        body.GetCollisionModel().AddCylinder(
            material, cylinder_r, depth / 2, depth / 2,
            chrono.ChVectorD(0, -length / 2 + gap_between_bodies + cylinder_r, 0),
            chrono.ChMatrix33D(chrono.Q_ROTATE_Z_TO_Y))

        body.GetCollisionModel().BuildModel()

        body.SetMass(mass)

        pos_in_marker = chrono.ChVectorD(0, -length / 2, 0)
        pos_out_marker = chrono.ChVectorD(0, length / 2, 0)
        super().__init__(builder,
                         body,
                         pos_in_marker,
                         pos_out_marker,
                         random_color,
                         is_collide=is_collide)


class FlatChronoBody(ChronoBody, RobotBody):
    """Class interprets node of robot flat (palm) in physic engine `pychrono <https://projectchrono.org/pychrono/>`_.
    
    Args:
        builder (chrono.ChSystem): Arg sets the system, which hosting the body
        length (float): Length of the robot link. Defaults to 2.
        width (float): Width of the robot link. Defaults to 0.1.
        depth (float): Height of the robot link. Defaults to 0.3.
        random_color (bool, optional): Flag of the random color of the body. Defaults to True.
        mass (float, optional): Value mass of the body box. Defaults to 1.
        material (Material, optional): Surface material, which define contact friction and etc.
        Defaults to DefaultChronoMaterial.
        is_collide (bool, optional): Flag of collision body with other object in system.
        Defaults to True.
    """

    def __init__(self,
                 builder,
                 length=2,
                 width=0.1,
                 depth=0.3,
                 random_color=True,
                 mass=1,
                 material=DefaultChronoMaterial(),
                 is_collide: bool = True):
        # Create body

        body = chrono.ChBody()

        box_asset = chrono.ChBoxShape()
        box_asset.GetBoxGeometry().Size = chrono.ChVectorD(width / 2, length / 2 - width / 32,
                                                           depth / 2)
        body.AddVisualShape(box_asset)
        body.SetCollide(True)

        body.SetMass(mass)

        pos_input_marker = chrono.ChVectorD(0, -length / 2, 0)
        pos_out_marker = chrono.ChVectorD(0, length / 2, 0)
        super().__init__(builder,
                         body,
                         pos_input_marker,
                         pos_out_marker,
                         random_color,
                         is_collide=is_collide)

        chrono_object_material = struct_material2object_material(material)

        self.body.GetCollisionModel().ClearModel()
        self.body.GetCollisionModel().AddBox(chrono_object_material, width / 2,
                                             length / 2 - width / 32, depth / 2)
        self.body.GetCollisionModel().SetDefaultSuggestedEnvelope(ENVELOPE)
        self.body.GetCollisionModel().SetDefaultSuggestedMargin(MARGIN)
        self.body.GetCollisionModel().BuildModel()


class MountChronoBody(ChronoBody, RobotBody):
    """Class is interprets node of robot end limbs in physic engine `pychrono <https://projectchrono.org/pychrono/>`_.
    
    Args:
        builder (chrono.ChSystem): Arg sets the system, which hosting the body
        length (float): Length of the robot link. Defaults to 0.1.
        width (float): Width of the robot link. Defaults to 0.1.
        depth (float): Height of the robot link. Defaults to 0.3.
        random_color (bool, optional): Flag of the random color of the body. Defaults to True.
        mass (float, optional): Value mass of the body box. Defaults to 1.
        material (Material, optional): Surface material, which define contact friction and etc.
        Defaults to DefaultChronoMaterial.
        is_collide (bool, optional): Flag of collision body with other object in system.
        Defaults to True.
    """

    def __init__(self,
                 builder,
                 length=0.1,
                 width=0.1,
                 depth=0.3,
                 random_color=True,
                 mass=1,
                 material=DefaultChronoMaterial(),
                 is_collide: bool = True):
        # Create body

        body = chrono.ChBody()

        box_asset = chrono.ChBoxShape()
        box_asset.GetBoxGeometry().Size = chrono.ChVectorD(width / 2, length / 2, depth / 2)
        body.AddVisualShape(box_asset)

        body.SetMass(mass)

        pos_input_marker = chrono.ChVectorD(0, -length / 2, 0)
        pos_out_marker = chrono.ChVectorD(0, length / 2, 0)
        super().__init__(builder,
                         body,
                         pos_input_marker,
                         pos_out_marker,
                         random_color,
                         is_collide=is_collide)

        chrono_object_material = struct_material2object_material(material)

        self.body.GetCollisionModel().ClearModel()
        self.body.GetCollisionModel().AddBox(chrono_object_material, width / 2, length / 2,
                                             depth / 2)
        self.body.GetCollisionModel().SetDefaultSuggestedEnvelope(ENVELOPE)
        self.body.GetCollisionModel().SetDefaultSuggestedMargin(MARGIN)
        self.body.GetCollisionModel().BuildModel()


class ChronoBodyEnv(ChronoBody):
    """Class of environments bodies with standard shape, like box, ellipsoid,
    cylinder. It adds solid body in `pychrono <https://projectchrono.org/pychrono/>`_ physical system that is not
    robot part.
    
    Args:
        builder (chrono.ChSystem): Arg sets the system, which hosting the body
        shape (SimpleBody): Args define the shape of the body. Defaults to SimpleBody.BOX
        random_color (bool, optional): Flag of the random color of the body. Defaults to True.
        mass (float, optional): Value mass of the body box. Defaults to 1.
        material (Material, optional): Surface material, which define contact friction and etc.
        Defaults to DefaultChronoMaterial.
        pos (FrameTransform): The frame define initial position and orientation .
    """

    def __init__(self,
                 builder,
                 shape=SimpleBody.BOX,
                 random_color=True,
                 mass=1,
                 material=DefaultChronoMaterial(),
                 pos: FrameTransform = FrameTransform([0, 0.0, 0], [1, 0, 0, 0])):

        # Create body
        material = struct_material2object_material(material)
        if shape is SimpleBody.BOX:
            body = chrono.ChBodyEasyBox(shape.value.width, shape.value.length, shape.value.height,
                                        1000, True, True, material)
        elif shape is SimpleBody.CYLINDER:
            body = chrono.ChBodyEasyCylinder(shape.value.radius, shape.value.height, 1000, True,
                                             True, material)
        elif shape is SimpleBody.SPHERE:
            body = chrono.ChBodyEasySphere(shape.value.radius, 1000, True, True, material)
        elif shape is SimpleBody.ELLIPSOID:
            body = chrono.ChBodyEasyEllipsoid(
                chrono.ChVectorD(shape.value.radius_a, shape.value.radius_b, shape.value.radius_c),
                1000, True, True, material)
        body.SetCollide(True)
        transform = ChronoTransform(builder, pos)
        body.SetCoord(transform.transform)
        body.GetCollisionModel().SetDefaultSuggestedEnvelope(ENVELOPE)
        body.GetCollisionModel().SetDefaultSuggestedMargin(MARGIN)
        body.SetMass(mass)

        # Create shape
        pos_in_marker = chrono.ChVectorD(0, 0, 0)
        pos_out_marker = chrono.ChVectorD(0, 0, 0)
        super().__init__(builder, body, pos_in_marker, pos_out_marker, random_color)

    def set_coord(self, frame: FrameTransform):
        transform = ChronoTransform(self.builder, frame)
        self.body.SetCoord(transform.transform)


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
                 builder: chrono.ChSystem,
                 axis: Axis = Axis.Z,
                 type_of_input: InputType = InputType.POSITION,
                 stiffness: float = 0.,
                 damping: float = 0.,
                 equilibrium_position: float = 0.):
        super().__init__(builder=builder)
        self.joint = None
        self.axis = axis
        self.input_type = type_of_input
        self._ref_frame_out = chrono.ChCoordsysD()
        # Spring Damper params
        self._joint_spring = None
        self._torque_functor = None
        self.stiffness = stiffness
        self.damping = damping
        self.equilibrium_position = equilibrium_position

    def connect(self, in_block: ChronoBody, out_block: ChronoBody):
        """Joint is connected two bodies.

        If we have two not initialize joints engine crash

        Args:
            in_block (ChronoBody): Slave body to connect
            out_block (ChronoBody): Master body to connect
        """
        self.joint = self.input_type.motor()
        self.joint.Initialize(in_block.body, out_block.body, True, in_block.transformed_frame_out,
                              out_block.ref_frame_in)
        self.builder.AddLink(self.joint)

        if (self.stiffness != 0) or (self.damping != 0):
            self._add_spring_damper(in_block, out_block)

    def apply_transform(self, in_block):
        """Aplied input tranformation to the output frame of the body.

        Args:
            in_block (BlockTransform): The block which define transormations
        """
        self.transformed_frame_out = self._ref_frame_out * in_block.transform

    def _add_spring_damper(self, in_block: ChronoBody, out_block: ChronoBody):
        self._joint_spring = chrono.ChLinkRSDA()
        self._joint_spring.Initialize(in_block.body, out_block.body, False,
                                      in_block.transformed_frame_out.GetAbsCoord(),
                                      out_block.ref_frame_in.GetAbsCoord())
        self._torque_functor = SpringTorque(self.stiffness, self.damping, self.equilibrium_position)
        self._joint_spring.RegisterTorqueFunctor(self._torque_functor)
        self.builder.Add(self._joint_spring)


class ChronoTransform(BlockTransform):
    """Class representing node of the transformation in `pychrono <https://projectchrono.org/pychrono/>`_ physical
    engine

    Args:
        builder (pychrono.ChSystem): Arg sets the system, which hosth the body
        transform (FrameTransform): Define tranformation of the instance
    """

    def __init__(self, builder: chrono.ChSystem, transform):
        super().__init__(builder=builder)
        if isinstance(transform, chrono.ChCoordsysD):
            self.transform = transform
        elif isinstance(transform, FrameTransform):
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
