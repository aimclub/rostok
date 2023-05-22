import pychrono as chrono
from example_vocabulary import (get_terminal_graph_no_joints, get_terminal_graph_three_finger,
                                get_terminal_graph_two_finger)

from rostok.block_builder_api.block_parameters import DefaultFrame, Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint

import numpy as np
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.simulation_chrono.basic_simulation import SystemPreviewChrono
from rostok.block_builder_chrono_alt.block_builder_chrono_api import ChronoBlockCreatorInterface as creator
from simple_designs import get_three_link_one_finger_with_no_control, get_two_link_one_finger
from rostok.library.rule_sets.ruleset_locomotion import get_bip, get_bip_single, get_box_joints
from rostok.graph_grammar.graph_utils import plot_graph, plot_graph_ids
mechs = [
    get_terminal_graph_three_finger, get_terminal_graph_no_joints, get_terminal_graph_two_finger
]
mechs = [get_two_link_one_finger
         #, get_three_link_one_finger_with_no_control
         ]
mechs = [get_box_joints]
def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]
for get_graph in mechs:

    graph = get_graph()
    sim = SystemPreviewChrono()

    # Create object to grasp
    mat = Material()
    mat.Friction = 0.65
    mat.DampingF = 0.65

    # obj = EnvironmentBodyBlueprint(shape = Box(3,0.2,3),material=mat,
    #                                pos=FrameTransform([0, -0.4, 0], [1,0,0,0]))
    # sim.add_object(creator.init_block_from_blueprint(obj))
    plot_graph(graph)
    sim.add_design(graph, FrameTransform([0, 2, 0], rotation_x(180)))
    sim.simulate(10000000000, True)
