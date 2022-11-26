import app_vocabulary

from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node import Node
from rostok.virtual_experiment.simulation_step import SimOut
import rostok.criterion.criterion_calc as criterion
from rostok.criterion.flags_simualtions import FlagMaxTime
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.trajectory_optimizer.trajectory_generator import create_torque_traj_from_x

import pychrono as chrono

def get_object_to_grasp():
    grab_obj_mat = chrono.ChMaterialSurfaceNSC()
    grab_obj_mat.SetFriction(0.5)
    grab_obj_mat.SetDampingF(0.1)
    obj = chrono.ChBodyEasyBox(0.2, 0.2, 0.6, 1000, True, True, grab_obj_mat)
    obj.SetCollide(True)
    obj.SetPos(chrono.ChVectorD(0, 1.2, 0))
    return obj


def grab_crtitrion(sim_output: dict[int, SimOut], grab_robot, node_feature: list[list[Node]], gait, weight):
    j_nodes = criterion.nodes_division(
        grab_robot, node_feature[1])
    b_nodes = criterion.nodes_division(
        grab_robot, node_feature[0])
    rb_nodes = criterion.sort_left_right(
        grab_robot, node_feature[3], node_feature[0])
    lb_nodes = criterion.sort_left_right(
        grab_robot, node_feature[2], node_feature[0])

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
    WEIGHT = [1, 1, 1, 1]

    cfg = ConfigRewardFunction()
    cfg.bound = (-5, 5)
    cfg.iters = 2
    cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
    cfg.time_step = 0.001
    cfg.time_sim = 2
    cfg.flags = [FlagMaxTime(2)]

    """Wraps function call"""

    criterion_callback = create_grab_criterion_fun(app_vocabulary.node_features, GAIT, WEIGHT)
    traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

    cfg.criterion_callback = criterion_callback
    cfg.get_rgab_object_callback = get_object_to_grasp
    cfg.params_to_timesiries_callback = traj_generator_fun

    control_optimizer = ControlOptimizer(cfg)
    graph = app_vocabulary.get_three_finger_graph()
    res = control_optimizer.start_optimisation(graph)
    print(res)
