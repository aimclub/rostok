from math import fabs
from turtle import shape
import app_vocabulary

from rostok.graph_grammar.node import GraphGrammar, BlockWrapper, Node
from rostok.graph_grammar.node import Node
from rostok.block_builder.node_render import *
from rostok.virtual_experiment.simulation_step import SimOut
import rostok.criterion.criterion_calc as criterion
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagNotContact
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.trajectory_optimizer.trajectory_generator import create_torque_traj_from_x
import matplotlib.pyplot as plt

import pychrono as chrono


def get_object_to_grasp():
    # grab_obj_mat = chrono.ChMaterialSurfaceNSC()
    # grab_obj_mat.SetFriction(0.5)
    # grab_obj_mat.SetDampingF(0.1)
    # obj = chrono.ChBodyEasyBox(0.2, 0.2, 0.6, 1000, True, True, grab_obj_mat)
    # obj.SetCollide(True)
    # obj.SetPos(chrono.ChVectorD(0, 1.2, 0))
    #obj = BlockWrapper(ChronoBodyEnv, width = 0.2, length = 0.2, pos=FrameTransform([0, 1, 0],[1,0,0,0]))
    #sliva
    matich = DefaultChronoMaterial()
    matich.Friction = 0.65
    matich.DampingF = 0.65
    obj = BlockWrapper(ChronoBodyEnv,
                       width=0.2,
                       length=0.3,
                       depth=0.5,
                       material=matich,
                       shape="ellipsoid",
                       pos=FrameTransform([0, 1, 0], [0, -0.048, 0.706, 0.706]))

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


if __name__ == '__main__':
    GAIT = 2.5
    WEIGHT = [5, 0, 1, 5]

    cfg = ConfigRewardFunction()
    cfg.bound = (-5, 5)
    cfg.iters = 5
    cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
    cfg.time_step = 0.001
    cfg.time_sim = 2
    cfg.flags = [FlagMaxTime(3)]
    """Wraps function call"""

    criterion_callback = create_grab_criterion_fun(app_vocabulary.node_features, GAIT, WEIGHT)
    traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

    cfg.criterion_callback = criterion_callback
    cfg.get_rgab_object_callback = get_object_to_grasp
    cfg.params_to_timesiries_callback = traj_generator_fun

    control_optimizer = ControlOptimizer(cfg)
    graph = app_vocabulary.get_three_finger_graph()

    res = [0, [0, 0, 0, 0, 0, 0]]
    print(res)
    myfun = control_optimizer.create_reward_function(graph)
    myfun(res[1], True)
