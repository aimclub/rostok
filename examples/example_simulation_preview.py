import numpy as np
import pychrono as chrono
from example_vocabulary import (get_terminal_graph_no_joints,
                                get_terminal_graph_three_finger,
                                get_terminal_graph_two_finger)

from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.block_parameters import (DefaultFrame,
                                                       FrameTransform,
                                                       Material)
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.graph_grammar.graph_utils import plot_graph, plot_graph_ids
from rostok.library.rule_sets.simple_designs import get_two_link_one_finger, get_palm
from rostok.simulation_chrono.basic_simulation import SystemPreviewChrono
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_obj_hard_mesh_piramida,get_object_easy_box, get_object_parametrized_box
mechs = [
    get_terminal_graph_three_finger, get_terminal_graph_no_joints, get_terminal_graph_two_finger
]
mechs = [get_two_link_one_finger
         #, get_three_link_one_finger_with_no_control
         ]
mechs = [get_palm]
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
    sim.add_design(graph, FrameTransform([0, 0, 0], rotation_x(0)))
    obj_bp = get_obj_hard_mesh_piramida()
    obj_bp = get_object_easy_box()
    # obj_bp =get_object_parametrized_box(0.3, 0.2, 0.4,2)
    sim.add_object(creator.create_environment_body(obj_bp))
    sim.simulate(10000000000, True)
