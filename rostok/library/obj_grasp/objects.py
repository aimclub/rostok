from rostok.block_builder_chrono import easy_body_shapes
from rostok.block_builder_chrono.block_classes import (ChronoEasyShapeObject,
                                              DefaultChronoMaterial,
                                              FrameTransform)
from rostok.virtual_experiment_chrono.flags_simualtions import FlagMaxTime
from rostok.graph.graph_utils import plot_graph
from rostok.graph.node import BlockWrapper
from rostok.trajectory_optimizer.control_optimizer import num_joints
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x
import numpy as np
import pychrono as chrono
from scipy.spatial.transform import Rotation

def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]


def rotation_z(alpha):
    quat_Z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
    return [quat_Z_ang_alpha.e0, quat_Z_ang_alpha.e1, quat_Z_ang_alpha.e2, quat_Z_ang_alpha.e3]


def rotation_y(alpha):
    quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
    return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2, quat_Y_ang_alpha.e3]

def get_obj_easy_box():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape_box = easy_body_shapes.Box(0.2, 0.2, 0.5)
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape_box,
                       material=matich,
                       pos=FrameTransform([0, 0.5, 0], [0, -0.048, 0.706, 0.706]))

    return obj

def get_obj_easy_large_box():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape_box = easy_body_shapes.Box(0.8, 0.4, 1.2)
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape_box,
                       material=matich,
                       pos=FrameTransform([0, 1, 0], [1, 0, 0, 0]))

    return obj

def get_obj_easy_long_box():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.1
    matich.DampingF = 0.65
    shape_box = easy_body_shapes.Box(0.4, 0.4, 4)
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape_box,
                       material=matich,
                       pos=FrameTransform([0, 0.5, 0], [1, 0, 0, 0]))

    return obj

def get_obj_easy_long_tilt_box():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape_box = easy_body_shapes.Box(0.4, 0.6, 4)
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape_box,
                       material=matich,
                       pos=FrameTransform([0, 1, 0], rotation_x(15)))

    return obj

def get_obj_easy_cylinder():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.2
    matich.DampingF = 0.65
    shape_box = easy_body_shapes.Cylinder(0.6, 0.4)
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape_box,
                       material=matich,
                       mass=1,
                       pos=FrameTransform([0, 1, 0], [1,0,0,0]))
    
    return obj

def get_object_to_grasp_sphere():
    """Medium task"""
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape = easy_body_shapes.Sphere(0.4)
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=matich,
                       pos=FrameTransform([0, 0.9, 0], [0, 0, 0, 1]))

    return obj



def get_obj_hard_mesh():
    # Create object to grasp
    shape = easy_body_shapes.FromMesh("examples\obj_grasp\Ocpocmaqs_scaled.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 1, 0], [ 0.854,  0.354,  0.354,  0.146]))
    return obj

def get_obj_hard_ellipsoid():
    shape = easy_body_shapes.Ellipsoid()
    shape.radius_x = 0.2
    shape.radius_y = 0.3
    shape.radius_z = 0.18
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 0.8, 0], [ 1,  0,  0, 0]))
    return obj


def get_obj_hard_large_ellipsoid():
    shape = easy_body_shapes.Ellipsoid()
    shape.radius_x = 0.35
    shape.radius_y = 0.5
    shape.radius_z = 0.4
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.6
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       mass = 1,
                       pos=FrameTransform([0.0, 0.8, 0.2], [ 1,  0,  0, 0]))
    return obj

def get_obj_hard_long_ellipsoid():
    shape = easy_body_shapes.Ellipsoid()
    shape.radius_x = 0.4
    shape.radius_y = 0.4
    shape.radius_z = 2
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 0.8, 0], [ 1,  0,  0, 0]))
    return obj

def get_obj_hard_long_tilt_ellipsoid():
    shape = easy_body_shapes.Ellipsoid()
    shape.radius_x = 0.4
    shape.radius_y = 0.4
    shape.radius_z = 2
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 0.8, 0], rotation_x(15)))
    return obj


def get_obj_box_pos_parametrize(yaw = 0 , pitch = 0, roll = 0):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = easy_body_shapes.Box(0.24, 0.24, 0.4)
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape_box,
                       material=mat,
                       pos=FrameTransform([0, 0.8, 0], quat))
    return obj

def get_obj_cyl_pos_parametrize(yaw = 0 , pitch = 0, roll = 0):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = easy_body_shapes.Cylinder()
    shape_box.height_y = 0.5
    shape_box.radius = 0.2
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape_box,
                       material=mat,
                       pos=FrameTransform([0, 0.8, 0], quat))
    return obj


def get_obj_hard_ellipsoid_move():
    shape = easy_body_shapes.Ellipsoid()
    shape.radius_x = 0.2
    shape.radius_y = 0.3
    shape.radius_z = 0.18
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0.15, 0.85, -0.1], [ 1,  0,  0, 0]))
    return obj



def get_obj_hard_mesh_bukvg():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("article\obj_grasp\G_BUKV_VERY2.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 1, 0], quat))
    return obj

def get_obj_hard_mesh_mikki():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("article\obj_grasp\MIKKI.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 1, 0], quat))
    return obj

def get_obj_hard_mesh_zateynik():
    # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("article\obj_grasp\ZATEYNIK.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 1, 0], quat))
    return obj

def get_obj_hard_mesh_piramida():
        # Create object to grasp
    quat = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
    shape = easy_body_shapes.FromMesh("article\obj_grasp\PIRAMIDA12.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 1, 0], quat))
    return obj
    