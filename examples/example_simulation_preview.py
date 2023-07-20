import numpy as np
import pychrono as chrono
from example_vocabulary import (get_terminal_graph_no_joints, get_terminal_graph_three_finger,
                                get_terminal_graph_two_finger)

from rostok.block_builder_api.block_parameters import FrameTransform
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.library.obj_grasp.objects import (get_obj_hard_mesh_piramida, get_object_easy_box,
                                              get_object_parametrized_box,
                                              get_object_parametrized_sphere)
from rostok.library.rule_sets.simple_designs import (get_palm, get_two_link_one_finger)
from rostok.simulation_chrono.basic_simulation import SystemPreviewChrono
from rostok.simulation_chrono.simulation_utils import \
    set_covering_sphere_based_position
from rostok.trajectory_optimizer.trajectory_generator import cable_length_linear_control, tendon_like_control


def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]


graph = get_terminal_graph_two_finger()
tendon = cable_length_linear_control(graph, [(0, 1), (0, 1)])
sim = SystemPreviewChrono()
sim.add_design(graph, FrameTransform([0, 0, 0], rotation_x(0)))
obj_bp = get_obj_hard_mesh_piramida()

grasp_object = creator.create_environment_body(obj_bp)
set_covering_sphere_based_position(grasp_object, reference_point=chrono.ChVectorD(0, 0.05, 0))
sim.add_object(grasp_object)
sim.simulate(10000000000, True)
