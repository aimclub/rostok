import numpy as np
import pychrono as chrono
from scipy.spatial.transform import Rotation

from rostok.block_builder_api import easy_body_shapes
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_chrono.block_classes import (DefaultChronoMaterial, FrameTransform)


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



def get_object_parametrized_box(x, y, z, h):
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape_box = easy_body_shapes.Box(x, y, z)
    object_blueprint = EnvironmentBodyBlueprint(shape=shape_box,
                                                material=matich,
                                                pos=FrameTransform([0, h, 0], [1, 0, 0, 0]))

    return object_blueprint


def get_object_parametrized_tilt_box(x, y, z, h, alpha):
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape_box = easy_body_shapes.Box(x, y, z)
    obj = EnvironmentBodyBlueprint(shape=shape_box,
                                   material=matich,
                                   pos=FrameTransform([0, h, 0], rotation_x(alpha)))

    return obj


def get_object_box_pos_parametrize(yaw=0, pitch=0, roll=0):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = easy_body_shapes.Box(0.24, 0.24, 0.4)

    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = EnvironmentBodyBlueprint(shape=shape_box,
                                   material=mat,
                                   pos=FrameTransform([0, 0.8, 0], quat))
    return obj


def get_object_parametrized_cylinder(radius, length, h):
    matich = DefaultChronoMaterial()
    matich.Friction = 0.2
    matich.DampingF = 0.65
    shape = easy_body_shapes.Cylinder(radius, length)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=matich,
                                   pos=FrameTransform([0, h, 0], [1, 0, 0, 0]))

    return obj


def get_obj_cyl_pos_parametrize(yaw=0, pitch=0, roll=0):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = easy_body_shapes.Cylinder()
    shape_box.height_y = 0.5
    shape_box.radius = 0.2
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = EnvironmentBodyBlueprint(shape=shape_box,
                                   material=mat,
                                   pos=FrameTransform([0, 0.8, 0], quat))

    return obj


def get_object_parametrized_sphere(r, h) -> EnvironmentBodyBlueprint:
    """Medium task"""
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape = easy_body_shapes.Sphere(r)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=matich,
                                   pos=FrameTransform([0, h, 0], [1, 0, 0, 0]))

    return obj


def get_object_parametrized_ellipsoid(x, y, z, h):
    shape = easy_body_shapes.Ellipsoid()
    shape.radius_x = x
    shape.radius_y = y
    shape.radius_z = z

    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=mat,
                                   pos=FrameTransform([0, h, 0], [1, 0, 0, 0]))
    return obj


def get_object_parametrized_tilt_ellipsoid(x, y, z, h, alpha):
    shape = easy_body_shapes.Ellipsoid()
    shape.radius_x = x
    shape.radius_y = y
    shape.radius_z = z

    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=mat,
                                   pos=FrameTransform([0, h, 0], rotation_x(alpha)))
    return obj


# special objects
def get_object_hard_mesh():
    # Create object to grasp
    shape = easy_body_shapes.FromMesh("examples\obj_grasp\Ocpocmaqs_scaled.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=mat,
                                   pos=FrameTransform([0, 1, 0], [0.854, 0.354, 0.354, 0.146]))

    return obj


def get_obj_hard_mesh_bukvg():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("article\obj_grasp\G_BUKV_VERY2.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat))
    return obj


def get_obj_hard_mesh_mikki():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("article\obj_grasp\MIKKI.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat))
    return obj


def get_obj_hard_mesh_zateynik():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("article\obj_grasp\ZATEYNIK.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat))
    return obj


def get_obj_hard_mesh_piramida():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("article\obj_grasp\PIRAMIDA12.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat))
    return obj


def get_obj_hard_get_obj_hard_large_ellipsoid():
    shape = easy_body_shapes.Ellipsoid()
    shape.radius_x = 0.5
    shape.radius_y = 1
    shape.radius_z = 0.6
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.5
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=mat,
                                   pos=FrameTransform([0.0, 1, 0.2], [1, 0, 0, 0]))
    return obj

def get_object_easy_box():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.2
    matich.DampingF = 0.65
    shape_box = easy_body_shapes.Box(0.4, 0.4, 0.7)
    object_blueprint = EnvironmentBodyBlueprint(shape=shape_box,
                                                material=matich,
                                                pos=FrameTransform([0, 0.5, 0],
                                                                   [0, -0.048, 0.706, 0.706]))

    return object_blueprint