import rostok.criterion.criterion_calc as criterion
import rostok.block_builder.body_size as bs
from rostok.block_builder.basic_node_block import SimpleBody
from rostok.block_builder.node_render import (ChronoBodyEnv,
                                              DefaultChronoMaterial,
                                              FrameTransform)
from rostok.graph_grammar.node import BlockWrapper, GraphGrammar, Node
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x
from rostok.virtual_experiment.simulation_step import SimOut
from rostok import intexp
import numpy as np


def get_object_to_grasp():
    matich = DefaultChronoMaterial()
    # matich.Friction = 0.65
    # matich.DampingF = 0.65
    obj = BlockWrapper(ChronoBodyEnv,
                       shape=SimpleBody.BOX,
                       material=matich,
                       pos=FrameTransform([0, 0.5, 0], [0, -0.048, 0.706, 0.706]))

    return obj

def get_pipes():

    # Create 3D mesh and setup parameters from files

    obj = BlockWrapper(ChronoBodyEnv)
    obj_db = intexp.chrono_api.ChTesteeObject()
    obj_db.create_chrono_body_from_file('./examples/models/custom/pipe_mul_10.obj',
                            './examples/models/custom/pipe.xml')
    new_gen_poses = intexp.poses_generator.gen_cylindrical_surface_around_object_axis(obj_db, 1, 0.3, 2.5, 'z')
    frame_1 = FrameTransform(new_gen_poses[1][0],[0,1,0,0])
    frame_2 = FrameTransform(new_gen_poses[3][0],[0,1,0,0])
    robot_frames = [frame_1, frame_2]
    return obj, robot_frames


def grab_crtitrion(sim_output: dict[int, SimOut], grab_robot, node_feature: list[list[Node]], gait,
                   weight):
    j_nodes = criterion.nodes_division(grab_robot, node_feature[1])
    b_nodes = criterion.nodes_division(grab_robot, node_feature[0])
    rb_nodes = criterion.sort_left_right(grab_robot, node_feature[3], node_feature[0])
    lb_nodes = criterion.sort_left_right(grab_robot, node_feature[2], node_feature[0])

    return criterion.criterion_calc(sim_output, b_nodes, j_nodes, rb_nodes, lb_nodes, weight, gait)


def create_grab_criterion_fun(node_features, gait, weight):

    def fun(sim_output, grab_robot):
        return grab_crtitrion(sim_output, grab_robot, node_features, gait, weight)

    return fun


def create_traj_fun(stop_time: float, time_step: float):

    def fun(graph: GraphGrammar, x: list[float]):
        return create_torque_traj_from_x(graph, x, stop_time, time_step)

    return fun
