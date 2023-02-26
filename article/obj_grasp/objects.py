from rostok.block_builder import envbody_shapes
from rostok.block_builder.node_render import (ChronoBodyEnv,
                                              DefaultChronoMaterial,
                                              FrameTransform)
from rostok.criterion.flags_simualtions import FlagMaxTime
from rostok.graph_grammar.graph_utils import plot_graph
from rostok.graph_grammar.node import BlockWrapper
from rostok.trajectory_optimizer.control_optimizer import num_joints
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x

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

def get_obj_easy_large_box():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape_box = envbody_shapes.Box(0.8, 0.4, 1.2)
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape_box,
                       material=matich,
                       pos=FrameTransform([0, 1, 0], [1, 0, 0, 0]))

    return obj


def get_obj_easy_long_tilt_box():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape_box = envbody_shapes.Box(0.4, 0.6, 4)
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape_box,
                       material=matich,
                       pos=FrameTransform([0, 1, 0], rotation_x(15)))

    return obj
def get_obj_easy_box():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape_box = envbody_shapes.Box(0.2, 0.2, 0.5)
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape_box,
                       material=matich,
                       pos=FrameTransform([0, 0.5, 0], [0, -0.048, 0.706, 0.706]))

    return obj



def get_object_to_grasp_sphere():    
    """Medium task"""
    matich = DefaultChronoMaterial()
    matich.Friction = 0.3
    matich.DampingF = 0.3
    shape = envbody_shapes.Sphere(0.3)
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape,
                       material=matich,
                       pos=FrameTransform([0, 0.9, 0], [0, 0, 0, 1]))

    return obj



def get_obj_hard_mesh():
    # Create object to grasp
    shape = envbody_shapes.FromMesh("examples\obj_grasp\Ocpocmaqs_scaled.obj")
    mat = DefaultChronoMaterial()
    mat.Friction = 0.2
    mat.DampingF = 0.2
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 1, 0], [ 0.854,  0.354,  0.354,  0.146]))
    return obj

def get_obj_hard_ellipsoid():
    shape = envbody_shapes.Ellipsoid()
    shape.radius_x = 0.2
    shape.radius_y = 0.3
    shape.radius_z = 0.18
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 0.8, 0], [ 1,  0,  0, 0]))
    return obj


def get_obj_hard_large_ellipsoid():
    shape = envbody_shapes.Ellipsoid()
    shape.radius_x = 0.4
    shape.radius_y = 0.4
    shape.radius_z = 0.6
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0, 0.8, 0], [ 1,  0,  0, 0]))
    return obj


def get_obj_box_pos_parametrize(yaw = 0 , pitch = 0, roll = 0):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = envbody_shapes.Box(0.24, 0.24, 0.4)
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape_box,
                       material=mat,
                       pos=FrameTransform([0, 0.8, 0], quat))
    return obj

def get_obj_cyl_pos_parametrize(yaw = 0 , pitch = 0, roll = 0):
    quat = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True).as_quat()
    shape_box = envbody_shapes.Cylinder()
    shape_box.height_y = 0.5
    shape_box.radius = 0.2
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape_box,
                       material=mat,
                       pos=FrameTransform([0, 0.8, 0], quat))
    return obj


def get_obj_hard_ellipsoid_move():
    shape = envbody_shapes.Ellipsoid()
    shape.radius_x = 0.2
    shape.radius_y = 0.3
    shape.radius_z = 0.18
    
    mat = DefaultChronoMaterial()
    mat.Friction = 0.30
    mat.DampingF = 0.8
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=shape,
                       material=mat,
                       pos=FrameTransform([0.15, 0.85, -0.1], [ 1,  0,  0, 0]))
    return obj