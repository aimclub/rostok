import numpy as np
import pychrono as chrono

import rostok.intexp as intexp
from rostok.graph_grammar.node import BlockWrapper
from rostok.block_builder.node_render import ChronoBodyEnv
from rostok.block_builder.transform_srtucture import FrameTransform
from rostok.block_builder.envbody_shapes import LoadedShape


def get_main_axis_pipe(obj_db):
    
    longest_axis_size = np.argmax(obj_db.bound_box)
    shorthest_axis_size = np.argmin(obj_db.bound_box)
    
    length_longest_dimension = obj_db.bound_box[longest_axis_size]
    length_shorthest_dimension = obj_db.bound_box[shorthest_axis_size]
    
    
    axis = {
        0: ('x', np.array([1, 0, 0])),
        1: ('y', np.array([0, 1, 0])),
        2: ('z', np.array([0, 0, 1]))
    }
    return (longest_axis_size,length_longest_dimension), (shorthest_axis_size,length_shorthest_dimension), axis

def create_builder_grab_object(path_to_pipe_obj = None,
                                path_to_pipe_xml = None):
    
    def get_pipe_object_n_pose():

        # Create 3D mesh and setup parameters from files

        obj = BlockWrapper(ChronoBodyEnv, LoadedShape(path_to_pipe_obj, path_to_pipe_xml))
        
        obj_db = intexp.entity.TesteeObject()
        obj_db.load_object_mesh(path_to_pipe_obj)
        obj_db.load_object_description(path_to_pipe_xml)
        
        long_dimension, short_dimension, axis = get_main_axis_pipe(obj_db)
        gen_poses = intexp.poses_generator.gen_cylindrical_surface_around_object_axis(
            obj_db, 1, short_dimension[1]*1.5,
            long_dimension[1] / 2, axis[long_dimension[0]][0])

        grab_pos = get_robot_poses_to_grasp_pipe(path_to_pipe_obj, path_to_pipe_xml)

        obj_db.clear_grasping_poses_list()
        obj_db.add_grasping_pose(grab_pos[0], [0, 0, 0, 1])
        obj_db.add_grasping_pose(grab_pos[1], [0, 0, 0, 1])
        
        center_pipe = FrameTransform(gen_poses[1][0],[0, 1, 0, 0])
        return obj, center_pipe
    
    return get_pipe_object_n_pose


def get_robot_poses_to_grasp_pipe(path_to_pipe_obj=None,
                                    path_to_pipe_xml=None):
    
    obj_db = intexp.chrono_api.ChTesteeObject()
    obj_db.create_chrono_body_from_file(path_to_pipe_obj, path_to_pipe_xml)
    long_dim, short_dim, axis =  get_main_axis_pipe(obj_db)

    gen_poses = intexp.poses_generator.gen_cylindrical_surface_around_object_axis(
        obj_db, 1, short_dim[1] * 1.5,
        long_dim[1]/2, axis[long_dim[0]][0])

    grab_pos_1 = gen_poses[1][0] + long_dim[1] * axis[long_dim[0]][1] * 0.25

    grab_pos_2 = gen_poses[1][0] - long_dim[1] * axis[long_dim[0]][1] * 0.25

    grab_frame_1 = FrameTransform(grab_pos_1, [0, 1, 0, 0])
    grab_frame_2 = FrameTransform(grab_pos_2, [0, 1, 0, 0])
    grab_poses = [grab_frame_1, grab_frame_2]
    return grab_poses