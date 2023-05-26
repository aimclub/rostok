import numpy as np

import pychrono as chrono
from example_vocabulary import (get_terminal_graph_three_finger, get_terminal_graph_two_finger, get_terminal_graph_no_joints)
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph

from rostok.block_builder_api.block_parameters import Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono
from rostok.block_builder_chrono_alt.block_builder_chrono_api import ChronoBlockCreatorInterface as creator
from rostok.library.rule_sets.simple_designs import  get_two_link_one_finger, get_one_link_one_finger
mechs = [get_terminal_graph_three_finger, get_terminal_graph_two_finger]
from rostok.graph_grammar.graph_utils import plot_graph, plot_graph_ids
from rostok.library.rule_sets.ruleset_locomotion import get_bip, get_bip_single, get_box, get_box_joints, get_box_one_joint

mechs = [get_one_link_one_finger]
mechs = [get_two_link_one_finger]
mechs = [get_bip]
def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]


for get_graph in mechs:

    graph = get_graph()
    # plot_graph(graph)
    # plot_graph_ids(graph)
    print(get_joint_vector_from_graph(graph))
    controll_parameters = []
    # for _ in range(len(get_joint_vector_from_graph(graph))):
    #     controll_parameters.append([2.0,np.random.normal(0,1,1)[0]])

    for _ in range(len(get_joint_vector_from_graph(graph))):
        controll_parameters.append(np.random.normal(0,1,1)[0])

    controll_parameters = {"initial_value": [0, 0, 0, 0, 0, 0 ,0 ,0, 0, 0], "sin_parameters" : [[0.000, 0.1, 0.1], [0.000, 0.1, 0.2],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1]]}
    #controll_parameters = {"initial_value": [0, 0, 0, 0, 0, 0 ,0 ,0], "sin_parameters" : [[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1]]}

    #controll_parameters = {"initial_value": [0], "sin_parameters" : [[0.001, 0.1, 0.1]]}
    print(controll_parameters)
    sim = RobotSimulationChrono([])
    # Create object to grasp
    mat = Material()
    mat.Friction = 0.65
    mat.DampingF = 0.65
    obj = EnvironmentBodyBlueprint(shape=Box(3, 0.2, 3),
                                   material=mat,
                                   pos=FrameTransform([0, 0, 0], [1, 0, 0, 0]))
    sim.add_object(creator.init_block_from_blueprint(obj))
    sim.objects[0].body.SetBodyFixed(True)
    sim.add_design(graph, controll_parameters, FrameTransform([0, 2.5, 0], rotation_x(180)), False)
    i=0
    # for idx, body in sim.robot.get_graph().body_map_ordered.items():
    #     if i == 2:
    #         body.body.SetBodyFixed(True)
    #     i+=1
    #print(sim.robot.sensor.joint_body_map)
    sim.simulate(1000, 0.01, 1, None, True)
    #print(sim.robot.sensor.trajectories)
    #print(sim.robot.data_storage.get_data("joint_trajectories"))
    for idx, value in sim.robot.data_storage.get_data('forces').items():
        print(idx)

    #print(sim.robot.data_storage.get_data('forces'))
