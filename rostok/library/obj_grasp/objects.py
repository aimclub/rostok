import numpy as np
import pychrono as chrono
from scipy.spatial.transform import Rotation

from rostok.block_builder_api import easy_body_shapes
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_chrono.block_classes import (DefaultChronoMaterialNSC, 
                                                       DefaultChronoMaterialSMC,
                                                       FrameTransform)


# rotation around axis with angle argument in degrees
def rotation_x(alpha):
    quat_x_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_x_ang_alpha.e0, quat_x_ang_alpha.e1, quat_x_ang_alpha.e2, quat_x_ang_alpha.e3]


def rotation_z(alpha):
    quat_z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
    return [quat_z_ang_alpha.e0, quat_z_ang_alpha.e1, quat_z_ang_alpha.e2, quat_z_ang_alpha.e3]


def rotation_y(alpha):
    quat_y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
    return [quat_y_ang_alpha.e0, quat_y_ang_alpha.e1, quat_y_ang_alpha.e2, quat_y_ang_alpha.e3]


# object functions return a blueprint of an object
def get_object_box(x, y, z, alpha):
    matich = DefaultChronoMaterialNSC()
    matich.Friction = 0.65
    matich.Restitution = 0.2
    shape_box = easy_body_shapes.Box(x, y, z)
    object_blueprint = EnvironmentBodyBlueprint(shape=shape_box,
                                                material=matich,
                                                pos=FrameTransform([0, 0, 0],
                                                                   rotation_x(alpha)))

    return object_blueprint

def get_object_box_rotation(x,y,z, yaw=0, pitch=0, roll=0):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = easy_body_shapes.Box(x, y, z)

    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.30
    mat.Restitution = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape_box,
                                   material=mat,
                                   pos=FrameTransform([0, 0, 0], quat))
    return obj


def get_object_cylinder(radius, length, alpha):
    matich = DefaultChronoMaterialNSC()
    matich.Friction = 0.2
    matich.Restitution = 0.2
    shape = easy_body_shapes.Cylinder(radius, length)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=matich,
                                   pos=FrameTransform([0, 0, 0], rotation_x(alpha)))

    return obj


def get_object_cylinder_rotation(radius, length, yaw=0, pitch=0, roll=0):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = easy_body_shapes.Cylinder()
    shape_box.height_y = length
    shape_box.radius = radius
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.30
    mat.Restitution = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape_box,
                                   material=mat,
                                   pos=FrameTransform([0, 0, 0], quat))

    return obj


def get_object_parametrized_sphere(r) -> EnvironmentBodyBlueprint:
    """Medium task"""
    matich = DefaultChronoMaterialNSC()
    matich.Friction = 0.65
    matich.Restitution = 0.2
    shape = easy_body_shapes.Sphere(r)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=matich,
                                   pos=FrameTransform([0, 0, 0], [1, 0, 0, 0]))

    return obj

def get_object_parametrized_sphere_smc(r) -> EnvironmentBodyBlueprint:
    """Medium task"""
    matich = DefaultChronoMaterialSMC()
    matich.Friction = 0.65
    shape = easy_body_shapes.Sphere(r)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=matich,
                                   pos=FrameTransform([0, 0, 0], [1, 0, 0, 0]))

    return obj


def get_object_ellipsoid(x, y, z, alpha):
    shape = easy_body_shapes.Ellipsoid()
    shape.radius_x = x
    shape.radius_y = y
    shape.radius_z = z

    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.30
    mat.DampingF = 0.5
    mat.Compliance = 0.0001
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=mat,
                                   pos=FrameTransform([0, 0, 0], rotation_x(alpha)),
                                   color=[215, 255, 0])
    return obj

# special objects
def get_object_hard_mesh():
    # Create object to grasp
    shape = easy_body_shapes.FromMesh("rostok\library\obj_grasp\Ocpocmaqs_scaled.obj")
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=mat,
                                   pos=FrameTransform([0, 1, 0], [0.854, 0.354, 0.354, 0.146]))

    return obj


def get_obj_hard_mesh_bukvg():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("rostok\library\obj_grasp\G_BUKV_VERY2.obj")
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat))
    return obj


def get_obj_hard_mesh_mikki():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("rostok\library\obj_grasp\MIKKI.obj")
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat))
    return obj


def get_obj_hard_mesh_zateynik():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("rostok\library\obj_grasp\ZATEYNIK.obj")
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat))
    return obj


def get_obj_hard_mesh_piramida():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("rostok\library\obj_grasp\PIRAMIDA12.obj")
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([-2, 1, 5], quat))
    return obj

