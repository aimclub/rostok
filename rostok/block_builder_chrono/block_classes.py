import pathlib
import random
from enum import Enum
from typing import Optional, Union

import open3d
import pychrono.core as chrono

import rostok.block_builder_api.easy_body_shapes as easy_body_shapes
from rostok.block_builder_api.block_parameters import (DefaultFrame, FrameTransform)
from rostok.block_builder_chrono.block_types import (BlockBody, BlockBridge, BlockTransform)
from rostok.block_builder_chrono.blocks_utils import (ContactReporter, SpringTorque,
                                                      frame_transform_to_chcoordsys, rotation_z_q)
from rostok.block_builder_chrono.mesh import o3d_to_chrono_trianglemesh
from rostok.utils.dataset_materials.material_dataclass_manipulating import (
    DefaultChronoMaterial, struct_material2object_material)


class BuildingBody(BlockBody):
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
                 color: Optional[list[int]] = None,
                 is_collide: bool = True):
        """Abstract class of interpretation of nodes of a robot body part in a
        physics engine.

        Initlization adds body in system, creates input and output
        marker of the body and sets them. Also, it initilize object of
        the contact reporter
        """
        super().__init__()
        # the body object is created in the constructor of the derived class
        self.body = body

        # Create markers - additional coordinate frames for the body
        # We add four markers 1. the default input connecting point, 2. the default out connecting point
        # 3. the transformed input point, 4. the transformed out point
        input_marker = chrono.ChMarker()
        out_marker = chrono.ChMarker()
        transformed_input_marker = chrono.ChMarker()
        transformed_out_marker = chrono.ChMarker()
        input_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
        out_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
        transformed_input_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
        transformed_out_marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
        self.body.AddMarker(input_marker)
        self.body.AddMarker(out_marker)
        self.body.AddMarker(transformed_input_marker)
        self.body.AddMarker(transformed_out_marker)
        input_marker.SetPos(in_pos_marker)
        out_marker.SetPos(out_pos_marker)
        transformed_input_marker.SetCoord(input_marker.GetCoord())
        transformed_out_marker.SetCoord(out_marker.GetCoord())
        self._ref_frame_in = input_marker
        self._ref_frame_out = out_marker
        self.transformed_frame_input = transformed_input_marker
        self.transformed_frame_out = transformed_out_marker

        # set the parameters of body collision model
        self.body.GetCollisionModel().SetDefaultSuggestedEnvelope(0.001)
        self.body.GetCollisionModel().SetDefaultSuggestedMargin(0.0005)
        self.body.SetCollide(is_collide)
        # Normal Forces
        # set a color for the body, default is random
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

    def apply_transform_out(self, transform: chrono.ChCoordsysD):
        """Applied transformation to the out frame of the body.

        Args:
            in_block (BlockTransform): The block which define transformations
        """
        self.reset_transformed_frame_out()
        frame_coord = self.transformed_frame_out.GetCoord()
        frame_coord = frame_coord * transform
        self.transformed_frame_out.SetCoord(frame_coord)

    def apply_input_transform(self, transform: chrono.ChCoordsysD):
        """Applied transformation to the input frame of the body.

        Args:
            in_block (BlockTransform): The block which define transformations
        """
        self.reset_transformed_frame_input()
        frame_coord = self.transformed_frame_input.GetCoord()
        frame_coord = frame_coord * transform
        self.transformed_frame_input.SetCoord(frame_coord)

    def set_coord(self, frame: FrameTransform):
        self.body.SetCoord(frame_transform_to_chcoordsys(frame))

    @property
    def ref_frame_in(self) -> chrono.ChMarker:
        """Return the input frame of the body.

        Returns:
            pychrono.ChMarker: The input frame of the body
        """
        return self._ref_frame_in


class ChronoTransform(BlockTransform):
    """Class representing node of the transformation in `pychrono <https://projectchrono.org/pychrono/>`_ physical
    engine

    Args:
        transform (FrameTransform): Define transformation of the instance
    """

    def __init__(self,
                 transform: Union[chrono.ChCoordsysD, FrameTransform],
                 is_transform_input=False):
        super().__init__(is_transform_input=is_transform_input)
        if isinstance(transform, chrono.ChCoordsysD):
            self.transform = transform
        elif isinstance(transform, FrameTransform):
            coordsys_transform = chrono.ChCoordsysD(
                chrono.ChVectorD(transform.position[0], transform.position[1],
                                 transform.position[2]),
                chrono.ChQuaternionD(transform.rotation[0], transform.rotation[1],
                                     transform.rotation[2], transform.rotation[3]))
            self.transform = coordsys_transform


class JointInputTypeChrono(str, Enum):
    TORQUE = {"Name": "Torque", "TypeMotor": chrono.ChLinkMotorRotationTorque}
    VELOCITY = {"Name": "Speed", "TypeMotor": chrono.ChLinkMotorRotationSpeed}
    POSITION = {"Name": "Angle", "TypeMotor": chrono.ChLinkMotorRotationAngle}
    UNCONTROL = {"Name": "Uncontrol", "TypeMotor": chrono.ChLinkRevolute}

    def __init__(self, vals):
        self.num = vals["Name"]
        self.motor = vals["TypeMotor"]


class ChronoRevolveJoint(BlockBridge):
    """The class represent revolute joint object in `pychrono <https://projectchrono.org/pychrono/>` 
    physical engine. It is the embodiment of joint nodes from the mechanism graph in
    simulation.

        Args:
            axis (Axis, optional): Define rotation axis. Defaults to Axis.Z.
            type_of_input (InputType, optional): Define type of input joint control. Defaults 
                to InputType.POSITION. Instead of, can changes to torque, that more realistic.
            stiffness (float, optional): Optional arg add a spring with `stiffness` to joint. 
                Defaults to 0.
            damping (float, optional): Optional arg add a damper to joint. Defaults to 0.
            equilibrium_position (float, optional): Define equilibrium position of the spring. 
                Defaults to 0.

        Attributes:
            joint (pychrono.ChLink): Joint define nodes of the joint part in the system
            axis (Axis): The axis of the rotation
            input_type (InputType): The type of input
    """

    def __init__(self,
                 type_of_input: JointInputTypeChrono = JointInputTypeChrono.TORQUE,
                 radius=0.07,
                 length=0.4,
                 material=DefaultChronoMaterial(),
                 density=100.0,
                 starting_angle=0,
                 stiffness: float = 0.,
                 damping: float = 0.,
                 equilibrium_position: float = 0.,
                 with_collision=True):
        super().__init__()
        self.joint: Optional[Union[chrono.ChLinkMotorRotationTorque,
                                   chrono.ChLinkMotorRotationSpeed, chrono.ChLinkMotorRotationAngle,
                                   chrono.ChLinkRevolute]] = None

        self.input_type = type_of_input
        self.radius = radius
        self.length = length
        self.starting_angle = starting_angle
        self.density = density
        material = struct_material2object_material(material)
        self.material = material
        # Spring Damper params
        self._joint_spring: Optional[chrono.ChLinkRSDA] = None
        self._torque_functor: Optional[SpringTorque] = None
        self.stiffness = stiffness
        self.damping = damping
        self.equilibrium_position = equilibrium_position
        self.with_collision = with_collision

    def set_prev_body_frame(self, prev_block: BuildingBody, system: chrono.ChSystem):
        # additional transform is just a translation along y axis to the radius of the joint
        additional_transform = chrono.ChCoordsysD(chrono.ChVectorD(0, self.radius, 0),
                                                  chrono.ChQuaternionD(1, 0, 0, 0))
        additional_transform *= chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0),
                                                   rotation_z_q(self.starting_angle))
        transform = prev_block.transformed_frame_out.GetCoord()
        prev_block.transformed_frame_out.SetCoord(transform * additional_transform)
        system.Update()

    def set_next_body_frame(self, next_block: BuildingBody, system: chrono.ChSystem):
        additional_transform = chrono.ChCoordsysD(chrono.ChVectorD(0, -self.radius, 0),
                                                  chrono.ChQuaternionD(1, 0, 0, 0))
        transform = next_block.transformed_frame_input.GetCoord()
        next_block.transformed_frame_input.SetCoord(transform * additional_transform)
        system.Update()

    def connect(self, in_block: BuildingBody, out_block: BuildingBody, system: chrono.ChSystem):
        """Joint is connected two bodies.

        If we have two not initialize joints engine crash

        Args:
            in_block (BuildingBody): Slave body to connect
            out_block (BuildingBody): Master body to connect
        """
        system.Update()
        self.joint = self.input_type.motor()
        self.joint.Initialize(in_block.body, out_block.body, True, in_block.transformed_frame_out,
                              out_block.transformed_frame_input)
        system.AddLink(self.joint)

        if (self.stiffness != 0) or (self.damping != 0):
            self._add_spring_damper(in_block, out_block, system)

        if self.with_collision:
            eps = 0.002
            cylinder = chrono.ChBodyEasyCylinder(self.radius - eps, self.length, self.density, True,
                                                 True, self.material)
            turn = chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.Q_ROTATE_Y_TO_Z)
            reversed_turn = chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.Q_ROTATE_Z_TO_Y)

            cylinder.SetCoord(in_block.transformed_frame_out.GetAbsCoord() * turn)
            system.Add(cylinder)
            marker = chrono.ChMarker()
            marker.SetMotionType(chrono.ChMarker.M_MOTION_KEYFRAMED)
            marker.SetCoord(reversed_turn)
            cylinder.AddMarker(marker)
            system.Update()
            fix_joint = chrono.ChLinkMateFix()
            fix_joint.Initialize(in_block.body, cylinder, True, in_block.transformed_frame_out,
                                 marker)
            system.Add(fix_joint)
        else:
            # Add cylinder visual only
            cylinder = chrono.ChCylinder()
            cylinder.p2 = chrono.ChVectorD(0, 0, self.length / 2)
            cylinder.p1 = chrono.ChVectorD(0, 0, -self.length / 2)
            cylinder.rad = self.radius
            cylinder_asset = chrono.ChCylinderShape(cylinder)
            self.joint.AddVisualShape(cylinder_asset)
        system.Update()

    def _add_spring_damper(self, in_block: BuildingBody, out_block: BuildingBody,
                           system: chrono.ChSystem):
        self._joint_spring = chrono.ChLinkRSDA()
        self._joint_spring.Initialize(in_block.body, out_block.body, False,
                                      in_block.transformed_frame_out.GetAbsCoord(),
                                      out_block.ref_frame_in.GetAbsCoord())
        self._torque_functor = SpringTorque(self.stiffness, self.damping, self.equilibrium_position)
        self._joint_spring.RegisterTorqueFunctor(self._torque_functor)
        system.Add(self._joint_spring)


class PrimitiveBody(BuildingBody):
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
                 shape: easy_body_shapes.ShapeTypes = easy_body_shapes.Box(),
                 density: float = 100.0,
                 material=DefaultChronoMaterial(),
                 is_collide: bool = True,
                 color: Optional[list[int]] = None):

        #offset
        eps = 0.001
        # Create body
        material = struct_material2object_material(material)

        if isinstance(shape, easy_body_shapes.Box):
            body = chrono.ChBodyEasyBox(shape.width_x, shape.length_y, shape.height_z, density,
                                        True, True, material)
            pos_in_marker = chrono.ChVectorD(0, -shape.length_y * 0.5 - eps, 0)
            pos_out_marker = chrono.ChVectorD(0, shape.length_y * 0.5 + eps, 0)
        elif isinstance(shape, easy_body_shapes.Cylinder):
            body = chrono.ChBodyEasyCylinder(shape.radius, shape.height_y, density, True, True,
                                             material)
            pos_in_marker = chrono.ChVectorD(0, -shape.height_y * 0.5 - eps, 0)
            pos_out_marker = chrono.ChVectorD(0, shape.height_y * 0.5 + eps, 0)
        elif isinstance(shape, easy_body_shapes.Sphere):
            body = chrono.ChBodyEasySphere(shape.radius, density, True, True, material)
            pos_in_marker = chrono.ChVectorD(0, -shape.radius * 0.5 - eps, 0)
            pos_out_marker = chrono.ChVectorD(0, shape.radius * 0.5 + eps, 0)
        elif isinstance(shape, easy_body_shapes.Ellipsoid):
            body = chrono.ChBodyEasyEllipsoid(
                chrono.ChVectorD(shape.radius_x, shape.radius_y, shape.radius_z), density, True,
                True, material)
            pos_in_marker = chrono.ChVectorD(0, -shape.radius_y * 0.5 - eps, 0)
            pos_out_marker = chrono.ChVectorD(0, shape.radius_y * 0.5 + eps, 0)
        else:
            raise Exception("Unknown shape for ChronoBodyEnv object")

        # Create shape
        super().__init__(body, pos_in_marker, pos_out_marker, color, is_collide)


class ChronoEasyShapeObject():
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
                 shape=easy_body_shapes.Box(),
                 density: float = 100.0,
                 material=DefaultChronoMaterial(),
                 is_collide: bool = True,
                 color: Optional[list[int]] = None,
                 pos: FrameTransform = DefaultFrame):

        # Create body
        material = struct_material2object_material(material)
        if isinstance(shape, easy_body_shapes.Box):
            body = chrono.ChBodyEasyBox(shape.width_x, shape.length_y, shape.height_z, density,
                                        True, True, material)
        elif isinstance(shape, easy_body_shapes.Cylinder):
            body = chrono.ChBodyEasyCylinder(shape.radius, shape.height_y, density, True, True,
                                             material)
        elif isinstance(shape, easy_body_shapes.Sphere):
            body = chrono.ChBodyEasySphere(shape.radius, density, True, True, material)
        elif isinstance(shape, easy_body_shapes.Ellipsoid):
            body = chrono.ChBodyEasyEllipsoid(
                chrono.ChVectorD(shape.radius_x, shape.radius_y, shape.radius_z), density, True,
                True, material)
        elif isinstance(shape, easy_body_shapes.FromMesh):
            if not pathlib.Path(shape.path).exists():
                raise Exception(f"Wrong path: {shape.path}")

            mesh = open3d.io.read_triangle_mesh(shape.path)
            mesh_chrono = o3d_to_chrono_trianglemesh(mesh)
            body = chrono.ChBodyEasyMesh(
                mesh_chrono,  # mesh filename
                density,  # density kg/m^3
                True,  # automatically compute mass and inertia
                True,  # visualize?>
                True,  # collide?
                material,  # contact material
            )
        else:
            raise Exception("Unknown shape for ChronoBodyEnv object")

        body.SetCoord(frame_transform_to_chcoordsys(pos))
        body.GetCollisionModel().SetDefaultSuggestedEnvelope(0.001)
        body.GetCollisionModel().SetDefaultSuggestedMargin(0.0005)
        body.SetCollide(is_collide)
        self.body = body
        if color is None:
            rgb = [random.random(), random.random(), random.random()]
            rgb[int(random.random() * 2)] *= 0.2
            self.body.GetVisualShape(0).SetColor(chrono.ChColor(*rgb))
        else:
            color = [x / 256 for x in color]
            self.body.GetVisualShape(0).SetColor(chrono.ChColor(*color))

    def set_coord(self, frame: FrameTransform):
        self.body.SetCoord(frame_transform_to_chcoordsys(frame))


BLOCK_CLASS_TYPES = Union[PrimitiveBody, ChronoRevolveJoint, ChronoTransform]
