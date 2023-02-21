from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact, FlagSlipout)
from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x, create_control_from_graph

from rostok.criterion.criterion_calc import criterion_calc
from functools import partial
from rostok.trajectory_optimizer.control_optimizer import ConfigConstTorque, ConfigGraphControl
from rostok.graph_grammar.node import Node

""" All config without get_rgab_object_callback"""


def get_cfg_standart():
    WEIGHT = [5, 5, 2]
    # Init configuration of control optimizing
    cfg = ConfigConstTorque()
    cfg.bound = (0, 15)
    cfg.iters = 20
    cfg.time_step = 0.0025
    cfg.time_sim = 3
    cfg.flags = [FlagMaxTime(cfg.time_sim), FlagNotContact(2), FlagSlipout(0.8, 0.8)]
    """Wraps function call"""
    criterion_callback = partial(criterion_calc, weights=WEIGHT)
    traj_generator_fun = partial(create_torque_traj_from_x,
                                 stop_time=cfg.time_sim,
                                 time_step=cfg.time_step)

    cfg.criterion_callback = criterion_callback
    cfg.params_to_timesiries_callback = traj_generator_fun
    return cfg


def get_cfg_graph(torque_dict: dict[Node, float]):
    WEIGHT = [5, 5, 2]
    # Init configuration of control optimizing
    cfg = ConfigGraphControl()
    cfg.time_step = 0.0025
    cfg.time_sim = 3
    cfg.flags = [FlagMaxTime(cfg.time_sim), FlagNotContact(2), FlagSlipout(0.8, 0.8)]
    """Wraps function call"""
    criterion_callback = partial(criterion_calc, weights=WEIGHT)
    traj_generator_fun = partial(create_control_from_graph,
                                 stop_time=cfg.time_sim,
                                 time_step=cfg.time_step,
                                 torque_dict=torque_dict)

    cfg.criterion_callback = criterion_callback
    cfg.params_to_timesiries_callback = traj_generator_fun
    return cfg