import rostok.criterion.criterion_calc as criterion
from rostok.block_builder_chrono.easy_body_shapes import Box
from rostok.block_builder_chrono.block_classes import (ChronoEasyShapeObject,
                                              DefaultChronoMaterial,
                                              FrameTransform)
from rostok.graph.node import BlockWrapper,  Node
from rostok.graph_grammar.graph_grammar import GraphGrammar
from rostok.trajectory_optimizer.trajectory_generator import create_torque_traj_from_x
from rostok.virtual_experiment.simulation_step import SimOut


def get_object_to_grasp():
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    shape_box = Box(0.2, 0.2, 0.4)
    obj = BlockWrapper(ChronoEasyShapeObject,
                       shape=shape_box,
                       material=matich,
                       pos=FrameTransform([0, 0.5, 0], [1, 0, 0, 0]))

    return obj


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
