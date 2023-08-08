import numpy as np

import pychrono as chrono
from example_vocabulary import (get_terminal_graph_three_finger, get_terminal_graph_two_finger,
                                get_terminal_graph_no_joints)
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph

from rostok.block_builder_api.block_parameters import Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono, SystemPreviewChrono
from rostok.block_builder_chrono.block_builder_chrono_api import ChronoBlockCreatorInterface as creator

from rostok.library.rule_sets.simple_designs_cm import get_two_link_one_finger, get_2_3_link_2_finger, get_one_link_one_finger

from rostok.graph_grammar.graph_utils import plot_graph, plot_graph_ids
from rostok.control_chrono.controller import ConstController, SinControllerChrono, YaxisShaker
from rostok.control_chrono.tendon_controller import PulleyParamsFinger_2p, TendonController_2p, create_pulley_params_finger_2p

mechs = [get_two_link_one_finger]


def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]


for get_graph in mechs:

    graph = get_graph()
    pp0 = PulleyParamsFinger_2p(0, (-0.01, -0.02, 0), (-0.01, 0.02, 0))
    pp1 = PulleyParamsFinger_2p(1, (0, -1, 2), (0, 1, 2))
    pp2 = PulleyParamsFinger_2p(2, (0, -1, 2), (0, 1, 2))
    pp3 = PulleyParamsFinger_2p(3, (0, -1, 2), (0, 1, 2))
    finger_parametrs_list = [pp0]
    kaifa3 = create_pulley_params_finger_2p(graph, finger_parametrs_list)
    controll_parameters = {
        "initial_value": [0, 0],
        "pulley_params_dict": kaifa3,
        "force_finger_dict": {
            0: 2
        }
    }
    print(controll_parameters)
    sim = RobotSimulationChrono([])
 
    # Create object to grasp
    mat = Material()
    mat.Friction = 0.65
    mat.DampingF = 0.65
    obj = EnvironmentBodyBlueprint(shape=Box(0.1, 0.07, 0.1),
                                   material=mat,
                                   pos=FrameTransform([0, -0.2, 0], [1, 0, 0, 0]))
    shake = YaxisShaker(25, 50)
    added_obj =  creator.init_block_from_blueprint(obj)
    sim.add_object(added_obj,
                   force_torque_controller=shake,
                   )

    sim.add_design(graph,
                   controll_parameters,
                   control_cls=TendonController_2p,
                   Frame=FrameTransform([0, 0, 0], rotation_x(180)),
                   is_fixed=True)
  
    sim.simulate(10000, 0.001, 1, None, True)
 
