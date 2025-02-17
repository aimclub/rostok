import numpy as np
import pychrono as chrono
from scipy.spatial.transform import Rotation

from rostok.block_builder_api import easy_body_shapes
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_chrono.block_classes import (
    DefaultChronoMaterialNSC, DefaultChronoMaterialSMC, FrameTransform)
from rostok.block_builder_chrono.block_comprehension import calc_volume_body



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

def get_object_sphere(r, mass=100, smc=False) -> EnvironmentBodyBlueprint:
    """Medium task"""
    if smc:
        mat = DefaultChronoMaterialSMC()
    else:
        mat = DefaultChronoMaterialNSC()
    density = mass / (4 / 3 * 3.14 * r**3)
    shape = easy_body_shapes.Sphere(r)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=mat,
                                   density=density,
                                   pos=FrameTransform([0, 0, 0], [1, 0, 0, 0]),
                                   color=[215, 255, 0])

    return obj


# object functions return a blueprint of an object
def get_object_box(x, y, z, alpha, mass):
    material = DefaultChronoMaterialNSC()
    material.Friction = 0.65
    material.DampingF = 0.65
    shape_box = easy_body_shapes.Box(x, y, z)
    object_blueprint = EnvironmentBodyBlueprint(shape=shape_box,
                                                material=material,
                                                pos=FrameTransform([0, 0, 0],
                                                                   rotation_x(alpha)), color=[215, 255, 0])

    volume = calc_volume_body(object_blueprint)
    object_blueprint.density  = mass / volume
    return object_blueprint

def get_object_box_rotation(x,y,z, yaw=0, pitch=0, roll=0):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = easy_body_shapes.Box(x, y, z)

    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = EnvironmentBodyBlueprint(shape=shape_box,
                                   material=mat,
                                   pos=FrameTransform([0, 0, 0], quat), color=[215, 255, 0])
    return obj


def get_object_cylinder(radius, length, alpha, mass):
    material = DefaultChronoMaterialNSC()
    material.Friction = 0.2
    material.DampingF = 0.65
    shape = easy_body_shapes.Cylinder(radius, length)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=material,
                                   pos=FrameTransform([0, 0, 0], rotation_x(alpha)), color=[215, 255, 0])
    volume = calc_volume_body(obj)
    obj.density  = mass / volume
    return obj


def get_object_cylinder_rotation(radius, length, yaw=0, pitch=0, roll=0, mass = 0.1):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = easy_body_shapes.Cylinder()
    shape_box.height_y = length
    shape_box.radius = radius
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = EnvironmentBodyBlueprint(shape=shape_box,
                                   material=mat,
                                   pos=FrameTransform([0, 0, 0], quat), color=[215, 255, 0])
    volume = calc_volume_body(obj)
    obj.density  = mass / volume
    return obj


def get_object_parametrized_sphere(r) -> EnvironmentBodyBlueprint:
    """Medium task"""
    material = DefaultChronoMaterialNSC()
    material.Friction = 0.65
    material.DampingF = 0.65
    shape = easy_body_shapes.Sphere(r)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=material,
                                   pos=FrameTransform([0, 0, 0], [1, 0, 0, 0]), color=[215, 255, 0])

    return obj


def get_object_ellipsoid(x, y, z, alpha, mass):
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
    volume = calc_volume_body(obj)
    obj.density  = mass / volume
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
                                   pos=FrameTransform([0, 1, 0], [0.854, 0.354, 0.354, 0.146]), color=[215, 255, 0])

    return obj


def get_obj_hard_mesh_bukvg():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("rostok\library\obj_grasp\G_BUKV_VERY2.obj")
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat), color=[215, 255, 0])
    return obj


def get_obj_hard_mesh_mikki():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("rostok\library\obj_grasp\MIKKI.obj")
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat), color=[215, 255, 0])
    return obj


def get_obj_hard_mesh_zateynik():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("rostok\library\obj_grasp\ZATEYNIK.obj")
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([0, 1, 0], quat), color=[215, 255, 0])
    return obj


def get_obj_hard_mesh_piramida():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("rostok\library\obj_grasp\PIRAMIDA12.obj")
    mat = DefaultChronoMaterialNSC()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = EnvironmentBodyBlueprint(shape=shape, material=mat, pos=FrameTransform([-2, 1, 5], quat), color=[215, 255, 0])
    return obj

def get_object_parametrized_cuboctahedron(a, mass = 0.100) -> EnvironmentBodyBlueprint:
    """Medium task"""
    material = DefaultChronoMaterialNSC()
    points = [(a, a, 0),
              (-a, a, 0),
              (a, -a, 0),
              (-a, -a, 0),

              (a, 0, a),
              (-a, 0 , a),
              (-a, 0, -a),
              (a, 0, -a),

              (0, a, a),
              (0, -a, a),
              (0, a, -a),
              (0, -a, -a)]
    shape = easy_body_shapes.ConvexHull(points)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=material,
                                   pos=FrameTransform([0, 0, 0], [1, 0, 0, 0]))
    volume = calc_volume_body(obj)
    obj.density  = mass / volume
    
    return obj



def get_object_parametrized_dipyramid_3(a, mass = 100) -> EnvironmentBodyBlueprint:
    """Medium task"""
    material = DefaultChronoMaterialNSC()
    C0 = np.sqrt(3) / 3
    C1 = 2 / 3
    C2 = 2 * np.sqrt(3) / 3

    V0 = (0.0, 0.0, a * C1)
    V1 = (0.0, 0.0, -C1*a)
    V2 = (1.0*a, a*C0, 0.0)
    V3 = (-1.0*a, a*C0, 0.0)
    V4 = (0.0, -C2*a, 0.0)
    points = [V0 ,V1, V2, V3, V4]
    shape = easy_body_shapes.ConvexHull(points)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=material,
                                   pos=FrameTransform([0, 0, 0], [1, 0, 0, 0]))
    volume = calc_volume_body(obj)
    obj.density  = mass / volume
    return obj


def get_object_parametrized_trapezohedron(a, mass = 100) -> EnvironmentBodyBlueprint:
    material = DefaultChronoMaterialNSC()
    C0 =  np.sqrt(2 * (3 * np.sqrt(2) - 4)) / 4
    C1 =   np.sqrt(2) / 2
    C2 =  np.sqrt(2 * (4 + 3 * np.sqrt(2))) / 4

    V0 = ( 0.0,  0.0,  C2*a)
    V1 = ( 0.0,  0.0, -C2*a)
    V2 = (  C1*a,  0.0,  C0*a)
    V3 = ( -C1*a,  0.0,  C0*a)
    V4 = ( 0.0,   C1*a,  C0*a)
    V5 = ( 0.0,  -C1*a,  C0*a)
    V6 = ( 0.5*a,  0.5*a, -C0*a)
    V7 = ( 0.5*a, -0.5*a, -C0*a)
    V8 = (-0.5*a,  0.5*a, -C0*a)
    V9 = (-0.5*a, -0.5*a, -C0*a)
    
    points = [V0 ,V1, V2, V3, V4, V5, V6, V7, V8, V9]
    shape = easy_body_shapes.ConvexHull(points)
    obj = EnvironmentBodyBlueprint(shape=shape,
                                   material=material,
                                   pos=FrameTransform([0, 0, 0], [1, 0, 0, 0]))
    volume = calc_volume_body(obj)
    obj.density  = mass / volume
    return obj